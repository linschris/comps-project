window.onload = () => {
    let foundHeader = document.getElementById("found-header")
    if(ytJSON != undefined) {
        let modelOptions = Object.keys(ytJSON)
        foundHeader.insertBefore(createDropdown(modelOptions), document.getElementById("found-header-title")) // Parent insert child before reference child
        insertYTEmbedLinks(modelOptions[0])
    }
}

function createDropdown(options) {
    // Create dropdown based on available models 
    let select = document.createElement('select')
    select.id = "model-dropdown"
    select.selectedIndex = 0 // First element is pre-selected by default
    select.addEventListener('change', (e) => updateYTEmbedLinks(e.target.value)) // When user change options on select, change items in youTubeDiv

    options.forEach(option => {
        let currOption = document.createElement('option')
        currOption.value = option
        currOption.textContent = option
        select.appendChild(currOption)
    })
    return select
}

function insertYTEmbedLinks(modelKey) {
    let currentVideos = ytJSON[modelKey]
    let youTubeDiv = document.getElementById("yt-embed-links")
    Object.keys(currentVideos).forEach(currVideoLink => {
        let videoDiv = document.createElement('div')
        videoDiv.classList.add('video-container')
        let currIFrame = document.createElement('iframe')
        currIFrame.classList.add('video-iframe')
        currIFrame.src = currVideoLink
        videoDiv.appendChild(currIFrame)
        if(currentVideos[currVideoLink].length > 0) {
            let relevantInfoContainer = document.createElement('div');
            relevantInfoContainer.classList.add('video-rel-info')
            let relevantInfoHeader = document.createElement('div');
            relevantInfoHeader.textContent = 'Relevant Times:'
            relevantInfoHeader.classList.add('subheading')
            relevantInfoContainer.appendChild(relevantInfoHeader)
            relevantInfoContainer.appendChild(createTimeDiv(currentVideos[currVideoLink], currIFrame))
            videoDiv.appendChild(relevantInfoContainer)
        }
        youTubeDiv.appendChild(videoDiv)
    })

}


function updateYTEmbedLinks(newKey) {
    // Remove current videos and replace with new YouTube links
    let youTubeDiv = document.getElementById("yt-embed-links")
    youTubeDiv.textContent = ''; // Remove all child nodes
    insertYTEmbedLinks(newKey)
}

function createTimeDiv(times, videoFrame) {
    let timeDiv = document.createElement('div')
    times.forEach(time => {
        let currTimeItem = document.createElement('button')
        currTimeItem.textContent = convertTimeFormat(time)
        currTimeItem.onclick = () => { videoFrame.src = `${videoFrame.src}?start=${time}&&autoplay=1` }
        timeDiv.appendChild(currTimeItem)
    })
    return timeDiv
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

