package com.micronaut.bug.config;

import io.micronaut.context.annotation.ConfigurationProperties;
import jakarta.annotation.PostConstruct;
import lombok.ToString;
import lombok.extern.slf4j.Slf4j;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.MonthDay;
import java.time.OffsetDateTime;
import java.time.OffsetTime;
import java.time.Period;
import java.time.Year;
import java.time.YearMonth;
import java.time.ZoneId;
import java.time.ZonedDateTime;

@Slf4j
//@Getter
//@Setter
@ToString
@ConfigurationProperties("app.entity")
public class MyEntityProperties {

    private Duration duration;
    private Period period;
    private YearMonth yearMonth;
    private Year year;
    private MonthDay monthDay;
    private ZoneId zoneId;
    private LocalDate localDate;
    private LocalTime localTime;
    private LocalDateTime localDateTime;
    private ZonedDateTime zonedDateTime;
    private OffsetDateTime offsetDateTime;
    private OffsetTime offsetTime;
    private Instant instant;

    @PostConstruct
    public void init() {
        log.info("config = {}", this);
    }

    public Duration getDuration() {
        return duration;
    }

    public void setDuration(Duration duration) {
        this.duration = duration;
    }

    public Period getPeriod() {
        return period;
    }

    public void setPeriod(Period period) {
        this.period = period;
    }

    public YearMonth getYearMonth() {
        return yearMonth;
    }

    public void setYearMonth(YearMonth yearMonth) {
        this.yearMonth = yearMonth;
    }

    public Year getYear() {
        return year;
    }

    public void setYear(Year year) {
        this.year = year;
    }

    public MonthDay getMonthDay() {
        return monthDay;
    }

    public void setMonthDay(MonthDay monthDay) {
        this.monthDay = monthDay;
    }

    public ZoneId getZoneId() {
        return zoneId;
    }

    public void setZoneId(ZoneId zoneId) {
        this.zoneId = zoneId;
    }

    public LocalDate getLocalDate() {
        return localDate;
    }

    public void setLocalDate(LocalDate localDate) {
        this.localDate = localDate;
    }

    public LocalTime getLocalTime() {
        return localTime;
    }

    public void setLocalTime(LocalTime localTime) {
        this.localTime = localTime;
    }

    public LocalDateTime getLocalDateTime() {
        return localDateTime;
    }

    public void setLocalDateTime(LocalDateTime localDateTime) {
        this.localDateTime = localDateTime;
    }

    public ZonedDateTime getZonedDateTime() {
        return zonedDateTime;
    }

    public void setZonedDateTime(ZonedDateTime zonedDateTime) {
        this.zonedDateTime = zonedDateTime;
    }

    public OffsetDateTime getOffsetDateTime() {
        return offsetDateTime;
    }

    public void setOffsetDateTime(OffsetDateTime offsetDateTime) {
        this.offsetDateTime = offsetDateTime;
    }

    public OffsetTime getOffsetTime() {
        return offsetTime;
    }

    public void setOffsetTime(OffsetTime offsetTime) {
        this.offsetTime = offsetTime;
    }

    public Instant getInstant() {
        return instant;
    }

    public void setInstant(Instant instant) {
        this.instant = instant;
    }
}
