function fetchSentiment() {
    const productSelect = document.getElementById('product_select');
    const productName = productSelect.value;
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    if (!productName) {
        results.innerHTML = '<p class="error">Please select a product.</p>';
        results.classList.remove('hidden');
        return;
    }

    loading.classList.remove('hidden');
    results.classList.add('hidden');

    fetch('/get_sentiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `product_name=${encodeURIComponent(productName)}`
    })
    .then(response => response.json())
    .then(data => {
        loading.classList.add('hidden');
        if (data.error || !data.sentiment) {
            results.innerHTML = `<p class="error">${data.error || 'No valid sentiment available'}</p>`;
        } else {
            results.innerHTML = `
                <div class="sentiment ${data.sentiment}">
                    ${data.sentiment.charAt(0).toUpperCase() + data.sentiment.slice(1)}
                </div>
            `;
        }
        results.classList.remove('hidden');
    })
    .catch(error => {
        loading.classList.add('hidden');
        results.innerHTML = '<p class="error">Error fetching sentiment. Please try again.</p>';
        results.classList.remove('hidden');
    });
}