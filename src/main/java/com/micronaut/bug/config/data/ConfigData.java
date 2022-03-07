package com.micronaut.bug.config.data;

import io.micronaut.core.annotation.Introspected;
import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.List;

/*
@Introspected
public record ConfigData(
        VariantEnum variant,
        List<String> params,
        List<Double> params2,
        List<ConfigItem> items
) {

    @Introspected
    public record ConfigItem(
            boolean prop1,
            double prop2
    ) {
    }
}
*/

@AllArgsConstructor
@Data
@Introspected
public class ConfigData {

    private VariantEnum variant;
    private List<String> params;
    private List<Double> params2;
    private List<ConfigItem> items;

    @AllArgsConstructor
    @Data
    @Introspected
    public static class ConfigItem {
        private boolean prop1;
        private double prop2;
    }
}
