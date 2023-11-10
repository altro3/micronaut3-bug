package com.micronaut.bug.config;

import jakarta.inject.Singleton;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;
import org.yaml.snakeyaml.nodes.MappingNode;
import org.yaml.snakeyaml.nodes.SequenceNode;

import jakarta.annotation.PostConstruct;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@Getter
@Singleton
public class YamlFileProperties {

    private Map<String, List<String>> props;

    @PostConstruct
    public void init() {

        // this logic works perfectly
        var yaml = new Yaml(new CustomSafeConstructor(new LoaderOptions()));

        var finalMap = new HashMap<String, List<String>>();

        Iterable<Object> objects = yaml.loadAll(getClass().getClassLoader().getResourceAsStream("properties.yml"));
        for (var object : objects) {
            if (object instanceof Map map) {
                for (var entryObj : map.entrySet()) {
                    var entry = (Map.Entry) entryObj;
                    finalMap.put(entry.getKey().toString(), (List<String>) entry.getValue());
                }
            }
        }

        props = finalMap;
        // this logic doesn't work
//        var yaml = new Yaml();
//        props = (Map<String, List<String>>) yaml.loadAs(getClass().getClassLoader().getResourceAsStream("properties.yml"), Map.class);
    }

    public List<String> getPropsByKey(String key) {
        return props.get(key);
    }

    private static class CustomSafeConstructor extends SafeConstructor {

        private CustomSafeConstructor(LoaderOptions loaderOptions) {
            super(loaderOptions);
        }

        @Override
        protected Map<Object, Object> newMap(MappingNode node) {
            return createDefaultMap(node.getValue().size());
        }

        @Override
        protected List<Object> newList(SequenceNode node) {
            return createDefaultList(node.getValue().size());
        }
    }

}
