// Video player implementation with proper cleanup
class VideoPlayer {
    constructor(videoElement, flvUrl) {
        this.videoElement = videoElement;
        this.flvUrl = flvUrl;
        this.flvPlayer = null;
        this.isSeeking = false;
        this.wasPlaying = false;
        this.boundHandleSeeking = this.handleSeeking.bind(this);
        this.boundHandleSeeked = this.handleSeeked.bind(this);

        // Initialize FLV player
        if (flvjs.isSupported()) {
            this.flvPlayer = flvjs.createPlayer({
                type: 'flv',
                url: this.flvUrl,
                isLive: true,
                hasAudio: true,
                hasVideo: true,
                enableStashBuffer: false,
                stashInitialSize: 128,
                enableWorker: true,
                lazyLoad: true,
                seekType: 'range'
            });

            this.flvPlayer.attachMediaElement(this.videoElement);
            this.flvPlayer.load();

            // Add event listeners
            this.videoElement.addEventListener('seeking', this.boundHandleSeeking);
            this.videoElement.addEventListener('seeked', this.boundHandleSeeked);
        }
    }

    handleSeeking() {
        this.isSeeking = true;
        this.wasPlaying = !this.videoElement.paused;
    }

    handleSeeked() {
        this.isSeeking = false;
        if (this.wasPlaying) {
            // Small delay to ensure the seek operation is complete
            setTimeout(() => {
                this.videoElement.play().catch(error => {
                    console.warn('Auto-resume after seeking failed:', error);
                });
            }, 100);
        }
    }

    play() {
        if (this.flvPlayer) {
            this.videoElement.play();
        }
    }

    pause() {
        if (this.flvPlayer) {
            this.videoElement.pause();
        }
    }

    destroy() {
        if (this.flvPlayer) {
            // Remove event listeners first
            this.videoElement.removeEventListener('seeking', this.boundHandleSeeking);
            this.videoElement.removeEventListener('seeked', this.boundHandleSeeked);

            // Pause the video
            this.videoElement.pause();

            // Unload the player
            this.flvPlayer.unload();
            this.flvPlayer.detachMediaElement();
            this.flvPlayer.destroy();
            this.flvPlayer = null;
        }
    }
}

// Export the VideoPlayer class
window.VideoPlayer = VideoPlayer; 