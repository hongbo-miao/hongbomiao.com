gradlew-version:
	./gradlew --version
gradlew-build:
	./gradlew build
gradlew-unit-test:
	./gradlew test
gradlew-instrumented-test:
	./gradlew connectedAndroidTest

adb-list:
	adb shell pm list instrumentation
adb-instrumented-test:
	adb shell am instrument -w com.example.hmmobile.test/androidx.test.runner.AndroidJUnitRunner
