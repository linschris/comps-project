class Prediction {
    // Creates a div to place and load the predictions and videos
    constructor(predictionJSON, parentDiv) {
        this.predictionJSON = predictionJSON
        this.parentDiv = parentDiv

        this.modelTypes = Object.keys(predictionJSON)
        this.predictions = predictionJSON[this.modelTypes[0]] // When first shown, show first model's predictions
        this.display()
    }

    createModelDropDown() {
        let select = document.createElement('select')
        select.classList.add("model-dropdown")
        select.selectedIndex = 0 // First model is pre-selected by default

         // When user change options on select, change items in youTubeDiv and reshow
        select.addEventListener('change', (e) => this.updatePredictions(e.target.value))

        this.modelTypes.forEach(model => {
            let currOption = document.createElement('option')
            currOption.value = model
            currOption.textContent = model
            select.appendChild(currOption)
        })
        return select
    }

    clearPredictions() {
        this.parentDiv.textContent = ''; // Remove all child nodes
    }

    updatePredictions(newModelKey) {
        // Clear, set new predictions, and display them
        this.clearPredictions()
        this.predictions = this.predictionJSON[newModelKey]
        this.display()
    }

    display() {
        let videoLinks = Object.keys(this.predictions)
        // Called when predictions should be shown
        videoLinks.forEach(videoLink => {
            // For each video link predicted, create a container to store the iframe and
            // relevant times, and append to the parent div.
            let currVideoContainer = this.createVideoContainer(videoLink)
            this.parentDiv.appendChild(currVideoContainer)
        })
    }

    createVideoContainer(currVideoLink) {
        // Create container to store both the video and relevant times in one section
        let videoDiv = document.createElement('div')
        videoDiv.classList.add('video-container')
        videoDiv.classList.add('flex-center-items')

        let currIFrame = document.createElement('iframe')
        currIFrame.classList.add('video-iframe')
        currIFrame.src = currVideoLink
        videoDiv.appendChild(currIFrame)

        let relevantTimes = this.predictions[currVideoLink]
        if(relevantTimes.length > 0) {
            // Create and append relevant time(s)
            let timeDiv = this.createTimeDiv(relevantTimes, currIFrame)
            videoDiv.appendChild(timeDiv)
        }
        return videoDiv
    }

    createTimeDiv(times, videoFrame) {
        let timeDiv = document.createElement('div')
        timeDiv.classList.add('video-relevant-times')
        times.forEach(time => {
            let currTimeItem = document.createElement('button')
            currTimeItem.textContent = convertTimeFormat(time)
            // Hacky, but will go to time in video upon click only using IFrames
            currTimeItem.onclick = () => { videoFrame.src = `${videoFrame.src}?start=${time}&&autoplay=1` }
            timeDiv.appendChild(currTimeItem)
        })
        return timeDiv
    }
}

function convertTimeFormat(seconds) {
    let numSeconds = parseInt(seconds);
    let numHours;
    let numMinutes;
    
    if (numSeconds > 3600) {
        numHours = Math.floor(numSeconds / 3600)
        numSeconds -= numHours * 3600
    }
    if (numSeconds > 60) {
        numMinutes = Math.floor(numSeconds / 60)
        numSeconds -= numMinutes * 60
    }
    let timeString = `${numHours > 0 ? numHours + ' hours' : ''} ${numMinutes > 0 ? numMinutes + ' minutes' : ''} ${numSeconds > 0 ? numSeconds + ' seconds' : ''}`
    return timeString
}