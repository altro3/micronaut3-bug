package com.micronaut.bug.config.data;

import io.micronaut.core.annotation.Introspected;
import io.micronaut.core.annotation.Nullable;

import java.util.List;

//@Serdeable
@Introspected
public record ConfigData(
        VariantEnum variant,
        List<String> params,
        List<Double> params2,
        List<ConfigItem> items
) {

    //    @Serdeable
    @Introspected
    public record ConfigItem(
            @Nullable Boolean prop1,
            @Nullable Double prop2
    ) {
    }
}

/*
@AllArgsConstructor
@Data
@Serdeable
public class ConfigData {

    private VariantEnum variant;
    private List<String> params;
    private List<Double> params2;
    private List<ConfigItem> items;

    @AllArgsConstructor
    @Data
    @Serdeable
    public static class ConfigItem {
        private boolean prop1;
        private double prop2;
    }
}
*/
