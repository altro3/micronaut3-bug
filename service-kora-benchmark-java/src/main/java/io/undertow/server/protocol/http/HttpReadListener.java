/*
 * JBoss, Home of Professional Open Source.
 * Copyright 2014 Red Hat, Inc., and individual contributors
 * as indicated by the @author tags.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package io.undertow.server.protocol.http;

import io.undertow.*;
import io.undertow.conduits.*;
import io.undertow.connector.*;
import io.undertow.protocols.http2.*;
import io.undertow.server.*;
import io.undertow.server.handlers.*;
import io.undertow.server.protocol.*;
import io.undertow.server.protocol.http2.*;
import io.undertow.util.*;
import org.xnio.*;
import org.xnio.channels.*;
import org.xnio.conduits.*;

import java.io.*;
import java.nio.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Listener which reads requests and headers off of an HTTP stream.
 *
 * @author <a href="mailto:david.lloyd@redhat.com">David M. Lloyd</a>
 */
final class HttpReadListener implements ChannelListener<ConduitStreamSourceChannel>, Runnable {

    private static final ThreadLocal<ArrayDeque<HttpServerExchange>> EXCHANGE_POOL =
        ThreadLocal.withInitial(() -> new ArrayDeque<>(16));

    /**
     * used for HTTP2 prior knowledge support
     */
    private static final HttpString PRI = new HttpString("PRI");
    private static final byte[] PRI_EXPECTED = new byte[] {'S', 'M', '\r', '\n', '\r', '\n'};


    private static final String BAD_REQUEST = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

    private final HttpServerConnection connection;
    private final RequestState state = new RequestState();
    private final RequestParser parser;

    private HttpServerExchange httpServerExchange;

    private int read = 0;
    private final int maxRequestSize;
    private final long maxEntitySize;
    private final boolean recordRequestStartTime;
    private final boolean allowUnknownProtocols;

    //0 = new request ok, reads resumed
    //1 = request running, new request not ok
    //2 = suspending/resuming in progress
    @SuppressWarnings("unused")
    private volatile int requestState;
    private static final AtomicIntegerFieldUpdater<HttpReadListener> requestStateUpdater = AtomicIntegerFieldUpdater.newUpdater(HttpReadListener.class, "requestState");

    private final ConnectorStatisticsImpl connectorStatistics;

    private ParseTimeoutUpdater parseTimeoutUpdater;

    HttpReadListener(final HttpServerConnection connection, final RequestParser parser, ConnectorStatisticsImpl connectorStatistics) {
        this.connection = connection;
        this.parser = parser;
        this.connectorStatistics = connectorStatistics;
        this.maxRequestSize = connection.getUndertowOptions().get(UndertowOptions.MAX_HEADER_SIZE, UndertowOptions.DEFAULT_MAX_HEADER_SIZE);
        this.maxEntitySize = connection.getUndertowOptions().get(UndertowOptions.MAX_ENTITY_SIZE, UndertowOptions.DEFAULT_MAX_ENTITY_SIZE);
        this.recordRequestStartTime = connection.getUndertowOptions().get(UndertowOptions.RECORD_REQUEST_START_TIME, false);
        this.allowUnknownProtocols = connection.getUndertowOptions().get(UndertowOptions.ALLOW_UNKNOWN_PROTOCOLS, UndertowOptions.DEFAULT_ALLOW_UNKNOWN_PROTOCOLS);
        int requestParseTimeout = connection.getUndertowOptions().get(UndertowOptions.REQUEST_PARSE_TIMEOUT, -1);
        int requestIdleTimeout = connection.getUndertowOptions().get(UndertowOptions.NO_REQUEST_TIMEOUT, -1);
        if (requestIdleTimeout < 0 && requestParseTimeout < 0) {
            this.parseTimeoutUpdater = null;
        } else {
            this.parseTimeoutUpdater = new ParseTimeoutUpdater(connection, requestParseTimeout, requestIdleTimeout);
            connection.addCloseListener(parseTimeoutUpdater);
        }
        if (connection.getUndertowOptions().contains(UndertowOptions.REQUIRE_HOST_HTTP11)) {
            UndertowLogger.ROOT_LOGGER.configurationNotSupported("REQUIRE_HOST_HTTP11");
        }
    }

    public void newRequest() {
        state.reset();
        read = 0;
        if (parseTimeoutUpdater != null) {
            parseTimeoutUpdater.connectionIdle();
        }
        connection.setCurrentExchange(null);
    }

    public void handleEvent(final ConduitStreamSourceChannel channel) {
        while (requestStateUpdater.get(this) != 0) {
            //if the CAS fails it is because another thread is in the process of changing state
            //we just immediately retry
            if (requestStateUpdater.compareAndSet(this, 1, 2)) {
                try {
                    channel.suspendReads();
                } finally {
                    requestStateUpdater.set(this, 1);
                }
                return;
            }
        }
        handleEventWithNoRunningRequest(channel);
    }

    public void handleEventWithNoRunningRequest(final ConduitStreamSourceChannel channel) {
        PooledByteBuffer existing = connection.getExtraBytes();
        if ((existing == null && connection.getOriginalSourceConduit().isReadShutdown()) || connection.getOriginalSinkConduit().isWriteShutdown()) {
            UndertowLogger.REQUEST_IO_LOGGER.debug("Connection is closing, cancelling handling of request");
            IoUtils.safeClose(connection);
            channel.suspendReads();
            return;
        }
        final PooledByteBuffer pooled;
        try {
            pooled = existing == null ? connection.getByteBufferPool().allocate() : existing;
        } catch (IllegalStateException e) {
            UndertowLogger.REQUEST_IO_LOGGER.debug("Connection is closing, cancelling handling of request", e);
            IoUtils.safeClose(connection);
            channel.suspendReads();
            return;
        }
        final ByteBuffer buffer = pooled.getBuffer();
        boolean free = true;

        try {
            int res;
            boolean bytesRead = false;
            do {
                if (existing == null) {
                    buffer.clear();
                    try {
                        res = channel.read(buffer);
                    } catch (IOException e) {
                        UndertowLogger.REQUEST_IO_LOGGER.debug("Error reading request", e);
                        IoUtils.safeClose(connection);
                        return;
                    }
                } else {
                    res = buffer.remaining();
                }

                if (res <= 0) {
                    if (bytesRead && parseTimeoutUpdater != null) {
                        parseTimeoutUpdater.failedParse();
                    }
                    handleFailedRead(channel, res);
                    return;
                } else {
                    bytesRead = true;
                }
                if (existing != null) {
                    existing = null;
                    connection.setExtraBytes(null);
                } else {
                    buffer.flip();
                }
                int begin = buffer.remaining();

                if (httpServerExchange == null) {
                    ArrayDeque<HttpServerExchange> pool = EXCHANGE_POOL.get();
                    httpServerExchange = pool.pollFirst();

                    if (httpServerExchange == null) {
                        httpServerExchange = new HttpServerExchange(connection, maxEntitySize);
                    } else {
                        httpServerExchange.updateConnectionState(connection, maxEntitySize);
                    }
                }

                parser.handle(buffer, state, httpServerExchange);
                if (state.isComplete()) {
                    if (!Cookies.isCrumbsAssemplyDisabled()) {
                        Cookies.assembleCrumbs(httpServerExchange.getRequestHeaders());
                    }
                }
                if (buffer.hasRemaining()) {
                    free = false;
                    connection.setExtraBytes(pooled);
                }
                int total = read + (begin - buffer.remaining());
                read = total;
                if (read > maxRequestSize) {
                    UndertowLogger.REQUEST_LOGGER.requestHeaderWasTooLarge(connection.getPeerAddress(), maxRequestSize);
                    sendBadRequestAndClose(connection.getChannel(), null);
                    return;
                }
            } while (!state.isComplete());
            if (parseTimeoutUpdater != null) {
                parseTimeoutUpdater.requestStarted();
            }
            connection.getOriginalSourceConduit().suspendReads();

            final HttpServerExchange httpServerExchange = this.httpServerExchange;
            httpServerExchange.setRequestScheme(connection.getSslSession() != null ? "https" : "http");
            this.httpServerExchange = null;
            requestStateUpdater.set(this, 1);

            if (recordRequestStartTime) {
                Connectors.setRequestStartTime(httpServerExchange);
            }

            if (httpServerExchange.getProtocol() == Protocols.HTTP_2_0) {
                free = handleHttp2PriorKnowledge(pooled, httpServerExchange);
                return;
            }

            if (!allowUnknownProtocols) {
                HttpString protocol = httpServerExchange.getProtocol();
                if (protocol != Protocols.HTTP_1_1 && protocol != Protocols.HTTP_1_0 && protocol != Protocols.HTTP_0_9) {
                    UndertowLogger.REQUEST_IO_LOGGER.debugf("Closing connection from %s due to unknown protocol %s", connection.getChannel().getPeerAddress(), protocol);
                    sendBadRequestAndClose(connection.getChannel(), new IOException());
                    return;
                }
            }
            HttpTransferEncoding.setupRequest(httpServerExchange);
            connection.setCurrentExchange(httpServerExchange);
            if (connectorStatistics != null) {
                connectorStatistics.setup(httpServerExchange);
            }
            if (connection.getSslSession() != null) {
                channel.suspendReads();
            }

            if (!Connectors.areRequestHeadersValid(httpServerExchange.getRequestHeaders())) {
                sendBadRequestAndClose(connection.getChannel(), UndertowMessages.MESSAGES.invalidHeaders());
                return;
            }

            Connectors.executeRootHandler(HostHeaderHandler.WRAPPER.wrap(connection.getRootHandler()), httpServerExchange);
        } catch (Throwable t) {
            sendBadRequestAndClose(connection.getChannel(), t);
            return;
        } finally {
            if (free) pooled.close();
        }
    }

    private boolean handleHttp2PriorKnowledge(PooledByteBuffer pooled, HttpServerExchange httpServerExchange) throws IOException {
        if (httpServerExchange.getRequestMethod().equals(PRI) && connection.getUndertowOptions().get(UndertowOptions.ENABLE_HTTP2, UndertowOptions.DEFAULT_ENABLE_HTTP2)) {
            handleHttp2PriorKnowledge(connection.getChannel(), connection, pooled);
            return false;
        } else {
            sendBadRequestAndClose(connection.getChannel(), new IOException());
            return true;
        }
    }

    private void handleFailedRead(ConduitStreamSourceChannel channel, int res) {
        if (res == 0) {
            channel.setReadListener(this);
            channel.resumeReads();
        } else if (res == -1) {
            IoUtils.safeClose(connection);
        }
    }

    private void sendBadRequestAndClose(final StreamConnection connection, final Throwable exception) {
        UndertowLogger.REQUEST_IO_LOGGER.failedToParseRequest(exception);
        connection.getSourceChannel().suspendReads();
        new StringWriteChannelListener(BAD_REQUEST) {
            @Override
            protected void writeDone(final StreamSinkChannel c) {
                super.writeDone(c);
                c.suspendWrites();
                IoUtils.safeClose(connection);
            }

            @Override
            protected void handleError(StreamSinkChannel channel, IOException e) {
                IoUtils.safeClose(connection);
            }
        }.setup(connection.getSinkChannel());
    }

    public void exchangeComplete(final HttpServerExchange exchange) {
        // === ШАГ 1: Запоминаем, можно ли переиспользовать объект ===
        // Делаем это строго в начале метода, пока Undertow не очистил стейт соединения
        final boolean canReuse = exchange.isPersistent() && !isUpgradeOrConnect(exchange);

        connection.clearChannel();
        connection.setCurrentExchange(null);
        final HttpServerConnection connection = this.connection;
        if (exchange.isPersistent() && !isUpgradeOrConnect(exchange)) {
            final StreamConnection channel = connection.getChannel();
            if (connection.getExtraBytes() == null) {
                if (exchange.isInIoThread()) {
                    newRequest();
                    channel.getSourceChannel().setReadListener(HttpReadListener.this);
                    channel.getSourceChannel().resumeReads();
                    requestStateUpdater.set(this, 0);
                } else {
                    while (true) {
                        if (connection.getOriginalSourceConduit().isReadShutdown() || connection.getOriginalSinkConduit().isWriteShutdown()) {
                            channel.getSourceChannel().suspendReads();
                            channel.getSinkChannel().suspendWrites();
                            IoUtils.safeClose(connection);
                            return;
                        } else {
                            if (requestStateUpdater.compareAndSet(this, 1, 2)) {
                                try {
                                    newRequest();
                                    channel.getSourceChannel().setReadListener(HttpReadListener.this);
                                    channel.getSourceChannel().resumeReads();
                                } finally {
                                    requestStateUpdater.set(this, 0);
                                }
                                break;
                            }
                        }
                    }
                }
            } else {
                if (exchange.isInIoThread()) {
                    requestStateUpdater.set(this, 0);
                    newRequest();
                    channel.getIoThread().execute(this);
                } else {
                    while (true) {
                        if (connection.getOriginalSinkConduit().isWriteShutdown()) {
                            channel.getSourceChannel().suspendReads();
                            channel.getSinkChannel().suspendWrites();
                            IoUtils.safeClose(connection);
                            return;
                        } else if (requestStateUpdater.compareAndSet(this, 1, 2)) {
                            try {
                                newRequest();
                                channel.getSourceChannel().suspendReads();
                            } finally {
                                requestStateUpdater.set(this, 0);
                            }
                            break;
                        }
                    }
                    Executor executor = exchange.getDispatchExecutor();
                    if (executor == null) {
                        executor = exchange.getConnection().getWorker();
                    }
                    executor.execute(this);
                }
            }
        } else if (!exchange.isPersistent()) {
            if (connection.getExtraBytes() != null) {
                connection.getExtraBytes().close();
                connection.setExtraBytes(null);
            }
            ConnectionUtils.cleanClose(connection.getChannel(), connection);
        } else {
            if (connection.getExtraBytes() != null) {
                connection.getChannel().getSourceChannel().setConduit(new ReadDataStreamSourceConduit(connection.getChannel().getSourceChannel().getConduit(), connection));
            }
            try {
                if (!connection.getChannel().getSinkChannel().flush()) {
                    connection.getChannel().getSinkChannel().setWriteListener(ChannelListeners.flushingChannelListener(conduitStreamSinkChannel -> connection.getUpgradeListener().handleUpgrade(connection.getChannel(), exchange), new ClosingChannelExceptionHandler<>(connection)));
                    connection.getChannel().getSinkChannel().resumeWrites();
                    return;
                }
                connection.getUpgradeListener().handleUpgrade(connection.getChannel(), exchange);
            } catch (IOException e) {
                UndertowLogger.REQUEST_IO_LOGGER.ioException(e);
                IoUtils.safeClose(connection);
            } catch (Throwable t) {
                UndertowLogger.REQUEST_IO_LOGGER.handleUnexpectedFailure(t);
                IoUtils.safeClose(connection);
            }
        }

        if (canReuse) {
            ArrayDeque<HttpServerExchange> pool = EXCHANGE_POOL.get();
            if (pool.size() < 32) {
                pool.offerLast(exchange);
            }
        }
    }

    private boolean isUpgradeOrConnect(HttpServerExchange exchange) {
        return exchange.isUpgrade() || (exchange.getRequestMethod().equals(Methods.CONNECT) && ((HttpServerConnection) exchange.getConnection()).isConnectHandled());
    }

    @Override
    public void run() {
        handleEvent(connection.getChannel().getSourceChannel());
    }


    private void handleHttp2PriorKnowledge(final StreamConnection connection, final HttpServerConnection serverConnection, PooledByteBuffer readData) throws IOException {

        final ConduitStreamSourceChannel request = connection.getSourceChannel();

        byte[] data = new byte[PRI_EXPECTED.length];
        final ByteBuffer buffer = ByteBuffer.wrap(data);
        if (readData.getBuffer().hasRemaining()) {
            while (readData.getBuffer().hasRemaining() && buffer.hasRemaining()) {
                buffer.put(readData.getBuffer().get());
            }
        }
        final PooledByteBuffer extraData;
        if (readData.getBuffer().hasRemaining()) {
            extraData = readData;
        } else {
            readData.close();
            extraData = null;
        }
        if (!doHttp2PriRead(connection, buffer, serverConnection, extraData)) {
            request.getReadSetter().set((ChannelListener<StreamSourceChannel>) channel -> {
                try {
                    doHttp2PriRead(connection, buffer, serverConnection, extraData);
                } catch (IOException e) {
                    UndertowLogger.REQUEST_IO_LOGGER.ioException(e);
                    IoUtils.safeClose(connection);
                } catch (Throwable t) {
                    UndertowLogger.REQUEST_IO_LOGGER.handleUnexpectedFailure(t);
                    IoUtils.safeClose(connection);
                }
            });
            request.resumeReads();
        }
    }

    private boolean doHttp2PriRead(StreamConnection connection, ByteBuffer buffer, HttpServerConnection serverConnection, PooledByteBuffer extraData) throws IOException {
        if (buffer.hasRemaining()) {
            int res = connection.getSourceChannel().read(buffer);
            if (res == -1) {
                return true; //fail
            }
            if (buffer.hasRemaining()) {
                return false;
            }
        }
        buffer.flip();
        for (int i = 0; i < PRI_EXPECTED.length; ++i) {
            if (buffer.get() != PRI_EXPECTED[i]) {
                throw UndertowMessages.MESSAGES.http2PriRequestFailed();
            }
        }

        Http2Channel channel = new Http2Channel(connection, null, serverConnection.getByteBufferPool(), extraData, false, false, false, serverConnection.getUndertowOptions());
        Http2ReceiveListener receiveListener = new Http2ReceiveListener(serverConnection.getRootHandler(), serverConnection.getUndertowOptions(), serverConnection.getBufferSize(), null);
        channel.getReceiveSetter().set(receiveListener);
        channel.resumeReceives();
        return true;
    }
}
