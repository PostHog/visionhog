document.addEventListener('DOMContentLoaded', function() {
    // Auto-refresh the stream list every 30 seconds
    if (window.location.pathname === '/') {
        setInterval(function() {
            fetch('/streams')
                .then(response => response.json())
                .then(data => {
                    if (data.streams.length !== document.querySelectorAll('.stream-card').length) {
                        // If the number of streams has changed, reload the page
                        window.location.reload();
                    }
                })
                .catch(error => console.error('Error fetching streams:', error));
        }, 30000);
    }
});