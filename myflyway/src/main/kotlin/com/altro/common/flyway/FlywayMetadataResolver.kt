package com.altro.common.flyway

import com.altro.common.flyway.FlywayConventionConst.CLEAN_VERSION_REGEX
import com.altro.common.flyway.FlywayConventionConst.DEFAULT_LOCATION
import com.altro.common.flyway.FlywayConventionConst.NO_ROLLBACK_SUFFIX
import com.altro.common.flyway.FlywayConventionConst.PREFIX_REPEATABLE
import com.altro.common.flyway.FlywayConventionConst.PREFIX_UNDO
import com.altro.common.flyway.FlywayConventionConst.PREFIX_VERSIONED
import com.altro.common.flyway.FlywayConventionConst.SEPARATOR_MIGRATION
import com.altro.common.flyway.FlywayConventionConst.SQL_ALL_PATTERN
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.boot.flyway.autoconfigure.FlywayProperties
import org.springframework.core.io.Resource
import org.springframework.core.io.ResourceLoader
import org.springframework.core.io.support.ResourcePatternUtils
import kotlin.time.measureTimedValue

class FlywayMetadataResolver(
    resourceLoader: ResourceLoader,
    private val flywayProperties: FlywayProperties
) {
    private val log = KotlinLogging.logger {}

    private val resourceResolver = ResourcePatternUtils.getResourcePatternResolver(resourceLoader)
    private var resolvedLocationsCache: Array<String>? = null

    // Выносим скомпилированный паттерн из цикла во внешний синглтон-объект или константу класса
    private val prefixTrimRegex = Regex("^[$PREFIX_VERSIONED$PREFIX_UNDO]")

    // Используем lazy (thread-safe по умолчанию), логика замера времени встроена внутрь инициализатора
    private val cachedMeta: MutableMap<String, MigrationResourceMeta> by lazy {
        val (metaMap, duration) = measureTimedValue {
            val internalMap = mutableMapOf<String, MigrationResourceMeta>()
            val locations = flywayProperties.locations.ifEmpty { listOf(DEFAULT_LOCATION) }

            val rawLocation = locations.firstOrNull() ?: DEFAULT_LOCATION
            val cleanBase = rawLocation.removePrefix("classpath:").removePrefix("/").removeSuffix("/")
            val baseMarker = "$cleanBase/"

            locations.forEach { rawLocationItem ->
                val cleanLocation = rawLocationItem.removeSuffix("/")
                val pattern = if (cleanLocation.contains(":")) "$cleanLocation/$SQL_ALL_PATTERN" else "classpath:$cleanLocation/$SQL_ALL_PATTERN"

                val resources = try {
                    resourceResolver.getResources(pattern)
                } catch (e: Exception) {
                    log.warn(e) { "Failed to scan location: $cleanLocation" }
                    emptyArray()
                }

                resources.forEach { resource ->
                    val filename = resource.filename ?: return@forEach

                    // Быстрая проверка строк вместо тяжелых операций
                    if (filename.startsWith(PREFIX_REPEATABLE)) {
                        throw IllegalStateException("Flyway validation failed! Repeatable migrations ('$PREFIX_REPEATABLE$SEPARATOR_MIGRATION') are prohibited.")
                    }

                    val isUndo = filename.startsWith(PREFIX_UNDO)
                    val isVersioned = filename.startsWith(PREFIX_VERSIONED)

                    if (!isVersioned && !isUndo) return@forEach

                    val cleanFilename = filename.replace(prefixTrimRegex, "")
                    val version = CLEAN_VERSION_REGEX.find(cleanFilename)?.value
                        ?: throw IllegalStateException("File '$filename' does not contain a valid timestamp prefix.")

                    if (version.length != 14) {
                        throw IllegalStateException("Invalid timestamp length in file '$filename'. Expected 14 digits.")
                    }

                    // Оптимизация: Избегаем тяжелого resource.url.toString(), используем дескриптор ресурса
                    val pathDescription = resource.description.replace("\\", "/")
                    val baseIndex = pathDescription.lastIndexOf(baseMarker)

                    val fullSubPath = if (baseIndex >= 0) {
                        val subPath = pathDescription.substring(baseIndex + baseMarker.length)
                        val lastSlash = subPath.lastIndexOf('/')
                        if (lastSlash >= 0) subPath.substring(0, lastSlash) else ""
                    } else {
                        ""
                    }

                    val cleanTag = valSeparatorIndex(fullSubPath)

                    val currentMeta = internalMap.computeIfAbsent(version) {
                        MigrationResourceMeta(
                            parentFolder = fullSubPath,
                            releaseTag = cleanTag,
                        )
                    }

                    if (isUndo) {
                        currentMeta.undoResource = resource
                    } else if (isVersioned) {
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

            // Финальная валидация консистентности пар
            internalMap.forEach { (version, meta) ->
                if (meta.forwardFilename != null && !meta.isNoRollback && meta.undoResource == null) {
                    throw IllegalStateException("Flyway validation failed! Missing undo partner for file '${meta.forwardFilename}'.")
                }
            }

            internalMap
        }

        // Логируем точное время сканирования и парсинга метаданных
        log.info { "Flyway metadata validation and scanning completed in $duration. Found ${metaMap.size} valid migration pairs." }
        metaMap
    }

    // Быстрый поиск последнего слэша без создания лишних подстрок через split/substringAfter
    private fun valSeparatorIndex(path: String): String {
        val index = path.lastIndexOf('/')
        return if (index >= 0) path.substring(index + 1) else path
    }

    fun getResolvedMetaMap(): Map<String, MigrationResourceMeta> = cachedMeta

    fun getResolvedLocations(): Array<String> {
        resolvedLocationsCache?.let { return it }

        val rawLocation = flywayProperties.locations.firstOrNull() ?: DEFAULT_LOCATION
        val cleanBase = rawLocation.removePrefix("classpath:").removePrefix("/").removeSuffix("/")

        // Оптимизация: Сразу резервируем емкость коллекции для уменьшения реалокаций памяти
        val activeSubDirs = HashSet<String>(cachedMeta.size)

        cachedMeta.values.forEach {
            if (it.parentFolder.isNotBlank()) {
                activeSubDirs.add("classpath:$cleanBase/${it.parentFolder}")
            } else {
                activeSubDirs.add("classpath:$cleanBase")
            }
        }

        val sortedSubDirs = activeSubDirs.sortedByDescending { it.length }
        val minimalLocations = mutableSetOf<String>()

        sortedSubDirs.forEach { location ->
            if (minimalLocations.none { it.startsWith("$location/") }) {
                minimalLocations.add(location)
            }
        }

        val locations = minimalLocations.toTypedArray()
        resolvedLocationsCache = locations
        return locations
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
