package com.micronaut.bug.flyway

import com.micronaut.bug.flyway.FlywayConventionConst.CLEAN_VERSION_REGEX
import com.micronaut.bug.flyway.FlywayConventionConst.DEFAULT_LOCATION
import com.micronaut.bug.flyway.FlywayConventionConst.NO_ROLLBACK_SUFFIX
import com.micronaut.bug.flyway.FlywayConventionConst.PREFIX_REPEATABLE
import com.micronaut.bug.flyway.FlywayConventionConst.PREFIX_UNDO
import com.micronaut.bug.flyway.FlywayConventionConst.PREFIX_VERSIONED
import com.micronaut.bug.flyway.FlywayConventionConst.SEPARATOR_MIGRATION
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.boot.autoconfigure.flyway.FlywayProperties
import org.springframework.core.io.Resource
import org.springframework.core.io.ResourceLoader
import org.springframework.core.io.support.ResourcePatternUtils

class FlywayMetadataResolver(
    resourceLoader: ResourceLoader,
    private val flywayProperties: FlywayProperties
) {
    private val log = KotlinLogging.logger {}
    private val resourceResolver = ResourcePatternUtils.getResourcePatternResolver(resourceLoader)

    // Превращаем мапу в классический MutableMap для возможности вызова .clear()
    private val cachedMeta: MutableMap<String, MigrationResourceMeta> by lazy {
        val metaMap = mutableMapOf<String, MigrationResourceMeta>()
        val locations = flywayProperties.locations.ifEmpty { listOf(DEFAULT_LOCATION) }

        // Вычисляем базовый маркер локации для честного отсечения подпутей
        val rawLocation = locations.firstOrNull() ?: DEFAULT_LOCATION
        val cleanBase = rawLocation.removePrefix("classpath:").removePrefix("/").removeSuffix("/")
        val baseMarker = "$cleanBase/"

        locations.forEach { rawLocationItem ->
            val cleanLocation = rawLocationItem.removeSuffix("/")
            val pattern = if (cleanLocation.contains(":")) "$cleanLocation/**/*.sql" else "classpath:$cleanLocation/**/*.sql"

            val resources = try {
                resourceResolver.getResources(pattern)
            } catch (e: Exception) {
                log.warn(e) { "Failed to scan location: $cleanLocation" }
                emptyArray()
            }

            resources.forEach { resource ->
                val filename = resource.filename ?: return@forEach
                if (filename.startsWith(PREFIX_REPEATABLE)) {
                    throw IllegalStateException("Flyway validation failed! Repeatable migrations ('$PREFIX_REPEATABLE$SEPARATOR_MIGRATION') are prohibited.")
                }

                if (!filename.startsWith(PREFIX_VERSIONED) && !filename.startsWith("U")) return@forEach

                val cleanFilename = filename.replace(Regex("^[$PREFIX_VERSIONED$PREFIX_UNDO]"), "")
                val version = CLEAN_VERSION_REGEX.find(cleanFilename)?.value
                    ?: throw IllegalStateException("File '$filename' does not contain a valid timestamp prefix.")

                if (version.length != 14) {
                    throw IllegalStateException("Invalid timestamp length in file '$filename'. Expected 14 digits.")
                }

                val urlPath = resource.url.toString().replace("\\", "/")
                val baseIndex = urlPath.lastIndexOf(baseMarker)

                val fullSubPath = if (baseIndex >= 0) {
                    val subPath = urlPath.substring(baseIndex + baseMarker.length)
                    if (subPath.contains("/")) subPath.substringBeforeLast("/") else ""
                } else {
                    ""
                }
                val cleanTag = if (fullSubPath.contains("/")) fullSubPath.substringAfterLast("/") else fullSubPath

                val currentMeta = metaMap.computeIfAbsent(version) {
                    MigrationResourceMeta(
                        parentFolder = fullSubPath,
                        releaseTag = cleanTag,
                    )
                }

                if (filename.startsWith(PREFIX_UNDO)) {
                    currentMeta.undoResource = resource
                } else if (filename.startsWith(PREFIX_VERSIONED)) {
                    if (filename.contains(NO_ROLLBACK_SUFFIX)) {
                        currentMeta.isNoRollback = true
                    }
                    if (currentMeta.forwardFilename != null) {
                        throw IllegalStateException("Duplicate version detected: '${currentMeta.forwardFilename}' and '$filename'")
                    }
                    currentMeta.forwardFilename = filename
                }
            }
        }

        metaMap.forEach { (version, meta) ->
            if (meta.forwardFilename != null && !meta.isNoRollback && meta.undoResource == null) {
                throw IllegalStateException("Flyway validation failed! Missing undo partner for file '${meta.forwardFilename}'.")
            }
        }

        metaMap
    }

    private var resolvedLocationsCache: Array<String>? = null

    fun getResolvedMetaMap(): Map<String, MigrationResourceMeta> = cachedMeta

    fun getResolvedLocations(): Array<String> {
        if (resolvedLocationsCache != null) return resolvedLocationsCache!!

        val rawLocation = flywayProperties.locations.firstOrNull() ?: DEFAULT_LOCATION
        val cleanBase = rawLocation.removePrefix("classpath:").removePrefix("/").removeSuffix("/")

        // Корректно склеиваем базовый путь и полный подпуть вложенности
        val activeSubDirs = cachedMeta.values.map {
            if (it.parentFolder.isNotBlank()) {
                "classpath:$cleanBase/${it.parentFolder}"
            } else {
                "classpath:$cleanBase"
            }
        }.toSet()

        val sortedSubDirs = activeSubDirs.sortedByDescending { it.length }
        val minimalLocations = mutableSetOf<String>()

        sortedSubDirs.forEach { location ->
            if (minimalLocations.none { it.startsWith("$location/") }) {
                minimalLocations.add(location)
            }
        }

        resolvedLocationsCache = minimalLocations.toTypedArray()
        return resolvedLocationsCache!!
    }

    fun clearAllMetadata() {
        log.info { "Purging heavy migration metadata RAM cache (Standard Startup Mode)..." }
        resolvedLocationsCache = null
        cachedMeta.clear()
    }

    class MigrationResourceMeta(
        var parentFolder: String,
        var releaseTag: String,
        var undoResource: Resource? = null,
        var isNoRollback: Boolean = false,
        var forwardFilename: String? = null
    )
}
