plugins {
    java
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.orekit:orekit:12.2")
    implementation("org.hipparchus:hipparchus-core:3.1")
    implementation("org.hipparchus:hipparchus-geometry:3.1")
    implementation("com.google.code.gson:gson:2.11.0")
}

java {
    // Use whatever JDK is available (17+ required by OreKit 12.x)
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

application {
    mainClass.set("benchmarks.Main")
}

tasks.named<JavaExec>("run") {
    standardInput = System.`in`
}
