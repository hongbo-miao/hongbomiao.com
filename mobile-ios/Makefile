build:
	xcodebuild build-for-testing \
		-project HMMobile.xcodeproj \
		-scheme HMMobile \
		-destination "platform=iOS Simulator,name=iPhone 14 Pro,OS=16.1"
test:
	xcodebuild test-without-building \
		-project HMMobile.xcodeproj \
		-scheme HMMobile \
		-destination "platform=iOS Simulator,name=iPhone 14 Pro,OS=16.1"
build-test:
	xcodebuild test \
		-project HMMobile.xcodeproj \
		-scheme HMMobile \
		-destination "platform=iOS Simulator,name=iPhone 14 Pro,OS=16.1"

bundle-init:
	bundle init
bundle-install:
	bundle install
bundle-exec-test-coverage:
	bundle exec slather
