package com.micronaut.bug.client

import io.micrometer.common.KeyValue
import io.micrometer.common.KeyValues
import org.springframework.http.client.observation.ClientRequestObservationContext
import org.springframework.http.client.observation.DefaultClientRequestObservationConvention

open class HttpClientObservationConvention(
    clientProperties: HttpClientProperties,
) : DefaultClientRequestObservationConvention() {

    private val serviceName: String? = clientProperties.serviceName ?: clientProperties.url.host

    override fun getLowCardinalityKeyValues(context: ClientRequestObservationContext): KeyValues {
        // Make sure that KeyValues entries are already sorted by name for better performance
        return KeyValues.of(clientName(context), exception(context), method(context), outcome(context), serviceName(context), status(context), uri(context))
    }

    private fun serviceName(context: ClientRequestObservationContext): KeyValue {
        if (context.carrier != null && serviceName != null) {
            return KeyValue.of(SERVICE_NAME, serviceName)
        }
        return SERVICE_NAME_NONE
    }

    companion object {

        private const val SERVICE_NAME = "service.name"
        private val SERVICE_NAME_NONE = KeyValue.of(SERVICE_NAME, KeyValue.NONE_VALUE)
    }
}
