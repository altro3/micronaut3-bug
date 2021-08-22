package com.micronaut.bug;

import io.micronaut.runtime.Micronaut;

public class Application {

    public static void main(String[] args) {

        Micronaut.build(args)
                .classes(Application.class)
                .banner(false)
                .start();
    }

}
