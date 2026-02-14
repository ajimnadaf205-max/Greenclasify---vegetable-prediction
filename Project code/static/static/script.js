// Navigation
const navLinks = document.querySelectorAll('.nav-link');
const pages = document.querySelectorAll('.page');
const pageTitle = document.getElementById('pageTitle');

function showPage(name){
  pages.forEach(p => { p.id === name ? p.removeAttribute('hidden') : p.setAttribute('hidden',''); });
  navLinks.forEach(n => n.classList.toggle('active', n.dataset.page === name));
  pageTitle.textContent = name.charAt(0).toUpperCase() + name.slice(1);
}

navLinks.forEach(link => link.addEventListener('click', ()=> showPage(link.dataset.page)));

// Prediction UI
const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const preview = document.getElementById('preview');
const resultCard = document.getElementById('resultCard');
const topPrediction = document.getElementById('topPrediction');
const nutritionDiv = document.getElementById('nutrition');
let chartInstance = null;

fileInput && fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  preview.src = url; preview.hidden = false; predictBtn.disabled = false; resultCard.hidden = true;
});

predictBtn && predictBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) return;
  predictBtn.disabled = true; predictBtn.textContent = 'Predicting...';

  const fd = new FormData(); fd.append('image', file);
  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (res.status !== 200) throw new Error(data.error || 'Prediction failed');

    // Show top prediction
    const pred = data.predictions[0];
    topPrediction.innerHTML = `<strong>${pred.class}</strong> — ${(pred.prob*100).toFixed(2)}%`;

    // Nutrition
    nutritionDiv.innerHTML = '';
    if (data.nutrition){
      const n = data.nutrition;
      nutritionDiv.innerHTML = `<strong>Nutrition (per 100g):</strong><div>Calories: ${n.Calories || '-'}<br>Vitamins: ${n.Vitamins || '-'}</div>`;
    }

    // Chart
    const labels = data.predictions.map(p => p.class);
    const values = data.predictions.map(p => +(p.prob*100).toFixed(2));
    const ctx = document.getElementById('chart').getContext('2d');
    if (chartInstance) chartInstance.destroy();
    chartInstance = new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets: [{ label: 'Confidence (%)', data: values, backgroundColor: '#60a5fa' }] },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { beginAtZero: true } },
        layout: { padding: 6 }
      }
    });

    resultCard.hidden = false;
    showPage('predict');
  } catch (err) {
    alert(err.message || 'Prediction error');
  } finally {
    predictBtn.disabled = false; predictBtn.textContent = 'Predict';
  }
});

// initial page
showPage('home');
