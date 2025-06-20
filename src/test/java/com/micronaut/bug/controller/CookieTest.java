package com.micronaut.bug.controller;

import io.micronaut.core.convert.ConversionService;
import io.micronaut.http.HttpHeaders;
import io.micronaut.http.MutableHttpRequest;
import io.micronaut.http.client.netty.NettyClientHttpRequestFactory;
import io.micronaut.http.cookie.Cookie;
import io.micronaut.http.netty.body.AvailableNettyByteBody;
import io.micronaut.http.server.HttpServerConfiguration;
import io.micronaut.http.server.netty.NettyHttpRequest;
import io.micronaut.http.simple.SimpleHttpRequestFactory;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpVersion;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.Mockito;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CookieTest {

    private static final Set<Cookie> COOKIES = new LinkedHashSet() {{
        add(Cookie.of("a", "1"));
        add(Cookie.of("b", "2"));
        add(Cookie.of("a", "3"));
    }};
    private static final String EXPECTED_VALUE = "a=3; b=2"; // Comment in NettyHttpClientRequest claims a space is needed :shrug:

    static List<Arguments> httpRequests() {
        MutableHttpRequest<?> simpleRequest = new SimpleHttpRequestFactory().get("/");
        MutableHttpRequest<?> nettyClientRequest = new NettyClientHttpRequestFactory().get("/");
        MutableHttpRequest<?> nettyHttpRequest = new NettyHttpRequest<>(
            new DefaultHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/"),
            AvailableNettyByteBody.empty(),
            Mockito.mock(ChannelHandlerContext.class),
            Mockito.mock(ConversionService.class),
            Mockito.mock(HttpServerConfiguration.class)).mutate();
        return List.of(
            Arguments.argumentSet(simpleRequest.getClass().getSimpleName(), simpleRequest),
            Arguments.argumentSet(nettyClientRequest.getClass().getSimpleName(), nettyClientRequest),
            Arguments.argumentSet(nettyHttpRequest.getClass().getSimpleName(), nettyHttpRequest)
        );
    }

    @ParameterizedTest
    @MethodSource("httpRequests")
    void requestCookiesOneByOne(MutableHttpRequest<?> request) {
        COOKIES.forEach(request::cookie);

        assertAll(
            () -> assertEquals(1, request.getHeaders().getAll(HttpHeaders.COOKIE).size()),
            () -> assertEquals(EXPECTED_VALUE, request.getHeaders().get(HttpHeaders.COOKIE))
        );
    }

    @ParameterizedTest
    @MethodSource("httpRequests")
    void requestCookiesBulk(MutableHttpRequest<?> request) {
        COOKIES.forEach(request::cookie);

        assertAll(
            () -> assertEquals(1, request.getHeaders().getAll(HttpHeaders.COOKIE).size()),
            () -> assertEquals(EXPECTED_VALUE, request.getHeaders().get(HttpHeaders.COOKIE))
        );
    }
}