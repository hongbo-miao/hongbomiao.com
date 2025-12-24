import QtQuick

Window {
    width: 640
    height: 480
    visible: true
    title: qsTr("Hongbo Miao")
    Text {
        id: slogan
        text: qsTr("Making magic happen")
        anchors.centerIn: parent
        font.bold: true
        font.pixelSize: 20
    }
}
