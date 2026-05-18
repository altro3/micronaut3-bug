package com.micronaut.bug.flyway

object FlywayConventionConst {
    // Префиксы и разделители Flyway
    const val PREFIX_VERSIONED = "V"
    const val PREFIX_UNDO = "U"
    const val PREFIX_REPEATABLE = "R"
    const val SEPARATOR_MIGRATION = "__"
    const val SEPARATOR_PLACEHOLDER = ":"

    // Плейсхолдеры
    const val PREFIX_PLACEHOLDER = $$"${"
    const val PREFIX_SCRIPT_PLACEHOLDER = $$"${"

    // Суффиксы и паттерны
    const val NO_ROLLBACK_SUFFIX = "_norb"
    const val SQL_ALL_PATTERN = "**/*.sql"

    // Дефолтные настройки
    const val DEFAULT_LOCATION = "classpath:db/migration"
    const val DEFAULT_TABLE = "flyway_schema_history"

    // Паттерн означает: игнорировать любые SQL-миграции, состояние которых оценивается как "missing" или "ignored"
    // Это заставит внутренности Flyway пропустить валидацию имен для файлов,
    // которые не участвуют в текущем цикле наката (включая ваши U-скрипты)
//    val CONFIG_IGNORE_PATTERNS = arrayOf("*:*", "versioned:missing", "versioned:ignored")
    val CONFIG_IGNORE_PATTERNS = arrayOf(
        "*:missing",  // Игнорировать, если запись в БД есть, а физического файла V в коде нет
        "*:ignored",  // Игнорировать пропущенные при накате
        "*:future"    // Игнорировать миграции из будущего
    )

    val CLEAN_VERSION_REGEX = Regex("""^\d+""")
}
