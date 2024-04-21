build:
	xcodebuild build-for-testing \
		-project HMMobile.xcodeproj \
		-scheme HMMobile \
		-destination "platform=iOS Simulator,name=iPhone 15 Pro,OS=17.4"
test:
	xcodebuild test-without-building \
		-project HMMobile.xcodeproj \
		-scheme HMMobile \
		-destination "platform=iOS Simulator,name=iPhone 15 Pro,OS=17.4"
build-test:
	xcodebuild test \
		-project HMMobile.xcodeproj \
		-scheme HMMobile \
		-destination "platform=iOS Simulator,name=iPhone 15 Pro,OS=17.4"

bundle-install:
	bundle install
bundle-add:
	bundle add xxx
bundle-update:
	bundle update
bundle-lock:
	bundle lock --add-platform x86_64-darwin-20
bundle-exec-test-coverage:
	bundle exec slather
