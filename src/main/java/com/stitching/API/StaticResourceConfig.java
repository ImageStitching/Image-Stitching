package com.stitching.API;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class StaticResourceConfig implements WebMvcConfigurer {
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        // Serve ảnh tại /images/** → từ thư mục uploads/
        registry.addResourceHandler("/stitch/**")
                .addResourceLocations("file:src/main/resources/stitch/");
    }
}