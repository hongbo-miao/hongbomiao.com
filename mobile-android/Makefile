gradlew-build:
	./gradlew build
gradlew-clean:
	./gradlew clean
gradlew-lint:
	./gradlew lint
gradlew-lint-fix:
	./gradlew lintFix
gradlew-unit-test:
	./gradlew test
gradlew-instrumented-test:
	./gradlew connectedAndroidTest
gradlew-list-tasks:
	./gradlew tasks
gradlew-version:
	./gradlew --version

lint-kotlin:
	./gradlew ktlintCheck
lint-kotlin-fix:
	./gradlew ktlintFormat
static-type-check-kotlin:
	./gradlew detekt

adb-list:
	adb shell pm list instrumentation
adb-instrumented-test:
	adb shell am instrument -w com.example.hmmobile.test/androidx.test.runner.AndroidJUnitRunner
