package com.micronaut.bug.config.data;

import io.micronaut.serde.annotation.Serdeable;

import java.util.List;

//@Introspected
@Serdeable
public record ConfigData(
        VariantEnum variant,
        List<String> params,
        List<Double> params2,
        List<ConfigItem> items
) {

    //    @Introspected
    @Serdeable
    public record ConfigItem(
            boolean prop1,
            double prop2
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
