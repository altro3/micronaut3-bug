package com.micronaut.bug.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import jakarta.annotation.PostConstruct;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import org.springframework.web.util.ContentCachingRequestWrapper;
import org.springframework.web.util.ContentCachingResponseWrapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

@Slf4j
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
@RequiredArgsConstructor
public class UnifiedLoggingFilter extends OncePerRequestFilter {

    private static final String X_REQ_ID = "x-req-id";
    private static final String BODY_EMPTY = "[Empty]";
    private static final String BODY_BINARY = "Binary data";
    private static final String BODY_NO_CONTENT = "No Body";
    private static final String STATUS_UNKNOWN = "UNKNOWN";
    private static final String PART_PREFIX = "  [PART] -> Name: %s | %s";
    private static final String PART_CONTENT = "Content: %s";
    private static final String PART_FILE = "File: %s";
    private static final String PART_EMPTY = "Empty content";

    private static final char SPACE = ' ';
    private static final char NEW_LINE = '\n';
    private static final String MINUS = "-";
    private static final String EMPTY_STRING = "";

    private static final String LOG_TEMPLATE_RQ = """
        
        ------------------ Service request ------------------
        URI: {} {}
        Headers: {}
        Body:
        {}
        ------------------ /Service request ------------------
        """;

    private static final String LOG_TEMPLATE_RS = """
        
        ------------------ Service response ------------------
        URI: {}
        Status: {}
        Headers: {}
        Body: {}
        ------------------ /Service response ------------------
        """;

    private static final int LIMIT_LOG_SIZE = 8192;

    private final ObjectMapper objectMapper;
    private ObjectMapper prettyMapper;

    @PostConstruct
    void init() {
        this.prettyMapper = objectMapper.copy().enable(SerializationFeature.INDENT_OUTPUT);
    }

    @Override
    protected void doFilterInternal(HttpServletRequest rq, HttpServletResponse rs, FilterChain chain)
        throws ServletException, IOException {

        var requestId = rq.getHeader(X_REQ_ID);
        if (requestId == null || requestId.isBlank()) {
            requestId = UUID.randomUUID().toString().replace(MINUS, EMPTY_STRING);
        }

        MDC.put(X_REQ_ID, requestId);

//        var rqWrapper = new ContentCachingRequestWrapper(rq);
//        var rsWrapper = new ContentCachingResponseWrapper(rs);

        try {
//            var isMultipart = isMultipart(rq);
//            if (isMultipart) {
//                logMultipartRequest(rqWrapper);
//            }

//            chain.doFilter(rqWrapper, rsWrapper);
            chain.doFilter(rq, rs);

//            if (!isMultipart) {
//                logSimpleRequest(rqWrapper);
//            }
//            logResponse(rqWrapper, rsWrapper);

        } finally {
//            rsWrapper.copyBodyToResponse();
            MDC.remove(X_REQ_ID);
        }
    }

    private void logSimpleRequest(ContentCachingRequestWrapper rq) {
        var content = rq.getContentAsByteArray();
        var bodyText = content.length == 0 ? BODY_NO_CONTENT : formatBody(content, rq.getContentType());
        log.info(LOG_TEMPLATE_RQ, rq.getMethod(), getFullUri(rq), getHeaders(rq), bodyText);
    }

    private void logMultipartRequest(HttpServletRequest rq) {
        var sb = new StringBuilder();
        try {
            var parts = rq.getParts();
            for (var part : parts) {
                if (!sb.isEmpty()) {
                    sb.append(NEW_LINE);
                }

                var fileName = part.getSubmittedFileName();
                var name = part.getName();

                if (fileName != null) {
                    sb.append(PART_PREFIX.formatted(name, PART_FILE.formatted(fileName)));
                } else {
                    try (var is = part.getInputStream()) {
                        var bytes = is.readAllBytes();
                        if (bytes.length == 0) {
                            sb.append(PART_PREFIX.formatted(name, PART_EMPTY));
                        } else if (isText(bytes)) {
                            var content = new String(bytes, StandardCharsets.UTF_8);
                            var formatted = formatIfJson(content, part.getContentType());
                            if (formatted.length() > LIMIT_LOG_SIZE) {
                                formatted = formatted.substring(0, LIMIT_LOG_SIZE) + "... [TRUNCATED]";
                            }

                            sb.append(PART_PREFIX.formatted(name, PART_CONTENT.formatted(formatted)));
                        } else {
                            sb.append(PART_PREFIX.formatted(name, BODY_BINARY));
                        }
                    }
                }
            }
        } catch (Exception e) {
            sb.append("[Multipart error: ").append(e.getMessage()).append(']');
        }
        var bodyResult = sb.isEmpty() ? BODY_NO_CONTENT : sb.toString();
        log.info(LOG_TEMPLATE_RQ, rq.getMethod(), getFullUri(rq), getHeaders(rq), bodyResult);
    }

    private void logResponse(HttpServletRequest rq, ContentCachingResponseWrapper rs) {
        var content = rs.getContentAsByteArray();
        var status = rs.getStatus() == 0 ? STATUS_UNKNOWN : String.valueOf(rs.getStatus());
        var bodyText = content.length == 0 ? BODY_EMPTY : formatBody(content, rs.getContentType());

        log.info(LOG_TEMPLATE_RS, getFullUri(rq), status, getResponseHeaders(rs), bodyText);
    }

    private String formatBody(byte[] content, String contentType) {
        if (!isText(content)) {
            return BODY_BINARY;
        }
        var raw = new String(content, StandardCharsets.UTF_8);
        return formatIfJson(raw, contentType);
    }

    private String formatIfJson(String body, String contentType) {
        if (body == null || body.isBlank()) {
            return EMPTY_STRING;
        }
        if (contentType != null && contentType.contains(MediaType.APPLICATION_JSON_VALUE)) {
            try {
                return prettyMapper.writerWithDefaultPrettyPrinter().writeValueAsString(prettyMapper.readTree(body));
            } catch (Exception ignored) {
            }
        }
        return body;
    }

    private String getHeaders(HttpServletRequest rq) {
        var sb = new StringBuilder().append('{');
        var names = rq.getHeaderNames();
        while (names.hasMoreElements()) {
            var name = names.nextElement();
            sb.append(name).append('=').append(rq.getHeader(name));
            if (names.hasMoreElements()) {
                sb.append(", ");
            }
        }
        return sb.append('}').toString();
    }

    private String getResponseHeaders(HttpServletResponse rs) {
        var sb = new StringBuilder().append('{');
        var names = rs.getHeaderNames();
        var it = names.iterator();
        while (it.hasNext()) {
            var name = it.next();
            sb.append(name).append('=').append(rs.getHeader(name));
            if (it.hasNext()) {
                sb.append(", ");
            }
        }
        return sb.append('}').toString();
    }

    private boolean isText(byte[] bytes) {
        for (int i = 0; i < Math.min(bytes.length, 100); i++) {
            if (bytes[i] == 0) {
                return false;
            }
        }
        return true;
    }

    private boolean isMultipart(HttpServletRequest rq) {
        var ct = rq.getContentType();
        return ct != null && ct.startsWith(MediaType.MULTIPART_FORM_DATA_VALUE);
    }

    private String getFullUri(HttpServletRequest rq) {
        var query = rq.getQueryString();
        return query == null ? rq.getRequestURI() : rq.getRequestURI() + '?' + query;
    }
}
