package com.altro.servicekora.service

import io.koraframework.resilient.retry.Retry
import io.koraframework.resilient.retry.annotation.RetrySpec

@RetrySpec("resilient.retry.postgres-retry")
interface PostgresRetry : Retry
