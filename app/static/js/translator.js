function translate() {
    const darijaText = encodeURIComponent(document.getElementById('darijaInput').value);
    fetch(`/translate/?darija_text=${darijaText}`, {
        method: 'GET',
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            document.getElementById('frenchTranslation').textContent = data.french;
            document.getElementById('englishTranslation').textContent = data.english;
        })
        .catch(error => {
            console.error('Error:', error);
        });
}