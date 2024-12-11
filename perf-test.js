
const username = 'avatar';
const password = 'avtarravi321';

// Encode username and password in Base64
const encodedCredentials = btoa(`${username}:${password}`);
// JavaScript
document.getElementById('startButton').addEventListener('click', startStreaming);

async function startStreaming() {
    try {
        const fileInput = document.getElementById('videoFileInput');
        const localVideo = document.getElementById('localVideo');
        const remoteVideosContainer = document.getElementById('remoteVideosContainer');
        const selectedFile = fileInput.files[0];
        const numConnections = parseInt(document.getElementById('connectionCount').value) || 1; // Get the number of connections from input


        if (!selectedFile) {
            console.error('No video file selected.');
            return;
        }

        // Load the video file into the video element
        const fileURL = URL.createObjectURL(selectedFile);
        localVideo.src = fileURL;

        // Play the video to ensure it's ready
        await localVideo.play();

        // Capture the video stream from the video element
        const stream = localVideo.captureStream();

        // Create multiple WebRTC connections
        for (let i = 0; i < numConnections; i++) {
            createPeerConnection(stream, i);
        }

        startButton.disabled = true;

    } catch (error) {
        console.error("Error starting stream:", error);
    }
}

async function createPeerConnection(stream, index) {
    try {
      const configuration = {
        sdpSemantics: 'unified-plan',
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' },
        ],
        iceCandidatePoolSize: 10
    }; // Add your STUN/TURN configuration here
        const pc = new RTCPeerConnection(configuration);

        // Create a new video element for the remote stream
        const remoteVideo = document.createElement('video');
        remoteVideo.autoplay = true;
        remoteVideo.controls = true;
        remoteVideo.id = `remoteVideo${index}`;
        document.getElementById('remoteVideosContainer').appendChild(remoteVideo);

        // Add local video stream to the connection
        pc.addTransceiver('video', {
            direction: 'sendrecv',
            streams: [stream],
        });
        pc.addTransceiver('audio', { direction: 'sendrecv' });

        stream.getTracks().forEach(track => pc.addTrack(track, stream));

        pc.oniceconnectionstatechange = () => {
            console.log("ICE connection state:", pc.iceConnectionState);
            if (pc.iceConnectionState === 'failed') {
                console.error('ICE connection failed');
            }
        };

        pc.onconnectionstatechange = () => {
            console.log("Connection state:", pc.connectionState);
            if (pc.connectionState === 'connected') {
                console.log('Connection established');
            }
        };

        pc.ontrack = (event) => {
            if (event.track.kind === 'video') {
                console.log(`Received video track for connection ${index}`, event.track);
                remoteVideo.srcObject = event.streams[0];
                event.track.onunmute = () => {
                    console.log(`Video track unmuted for connection ${index} and ready to play`);
                };
            }
        };

        // Create an offer and set up the connection
        const offer = await pc.createOffer({
            offerToReceiveAudio: true,
            offerToReceiveVideo: true
        });

        offer.sdp = offer.sdp.replace("a=mid:video", "a=mid:video\r\na=jitterMinDelay:0");
        offer.sdp = preferCodec(offer.sdp, 'video', 'H264');

        await pc.setLocalDescription(offer);

        // Send the offer to your server and get an answer
        const response = await fetch('/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json',
              'Authorization': `Basic ${encodedCredentials}`
             },
            body: JSON.stringify({
                sdp: pc.localDescription.sdp,
                type: pc.localDescription.type,
            }),
        });

        const answer = await response.json();
        await pc.setRemoteDescription(answer);

    } catch (error) {
        console.error(`Error creating peer connection ${index}:`, error);
    }
}

function preferCodec(sdp, media, codec) {
    // Function to prefer a specific codec in SDP
    const lines = sdp.split('\n');
    let mLineIndex = -1;
    let codecIndex = -1;

    for (let i = 0; i < lines.length; i++) {
        if (lines[i].startsWith(`m=${media}`)) {
            mLineIndex = i;
        }
        if (mLineIndex !== -1 && lines[i].includes(codec)) {
            codecIndex = i;
            break;
        }
    }

    if (mLineIndex !== -1 && codecIndex !== -1) {
        const mLine = lines[mLineIndex].split(' ');
        const codecNumber = lines[codecIndex].match(/(\d+)/)[0];
        mLine.splice(3, 0, codecNumber);
        lines[mLineIndex] = mLine.join(' ');
    }

    return lines.join('\n');
}
