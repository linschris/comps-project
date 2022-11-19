window.onload = () => {
    if(ytJSON != undefined) {
        let foundHeader = document.getElementById("found-header")
        let youTubeDiv = document.getElementById("yt-embed-links")
        let prediction = new Prediction(ytJSON, youTubeDiv) // creates prediction div and appends to youTubeDiv
        foundHeader.insertBefore(prediction.createModelDropDown(prediction.modelTypes), document.getElementById("found-header-title"))
    }
}