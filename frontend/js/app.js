function switchTab(tabName) {
    ['overview', 'simulator', 'analytics'].forEach(tab => {
        document.getElementById(`tab-${tab}`).classList.add('hidden');
        document.getElementById(`btn-${tab}`).classList.remove('text-brand-500', 'bg-slate-800');
        document.getElementById(`btn-${tab}`).classList.add('text-slate-300');
    });

    document.getElementById(`tab-${tabName}`).classList.remove('hidden');
    const activeBtn = document.getElementById(`btn-${tabName}`);
    activeBtn.classList.add('text-brand-500', 'bg-slate-800');
    activeBtn.classList.remove('text-slate-300');

    if (tabName === 'analytics' && !window.chartsRendered) {
        renderCharts();
        window.chartsRendered = true;
    }
}

function updateInputVal(id, val, suffix = '') {
    document.getElementById(`val-${id}`).innerText = val + suffix;
}

async function runSimulation() {
    const payload = {
        density: parseFloat(document.getElementById('density').value),
        gravity: parseFloat(document.getElementById('gravity').value),
        purity: parseFloat(document.getElementById('purity').value),
        refractive: parseFloat(document.getElementById('refractive').value),
        crystal: document.getElementById('crystal-system').value,
        region: document.getElementById('region').value
    };

    try {
        const response = await fetch('http://localhost:5000/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Server prediction failed');

        const result = await response.json();

        document.getElementById('predicted-score').innerText = result.predicted_hardness + ' / 10';
        document.getElementById('m-hardness').innerText = result.predicted_hardness;
        document.getElementById('m-recovery').innerText = result.recovery_rate;

        const badgeElem = document.getElementById('quality-badge');
        badgeElem.innerText = result.quality_grade;

        if (result.quality_grade === 'Low Quality') {
            badgeElem.className = 'inline-block px-4 py-1.5 rounded-full text-sm font-extrabold uppercase tracking-wider bg-rose-500/20 text-rose-400 border border-rose-500/30';
        } else if (result.quality_grade === 'Medium Quality') {
            badgeElem.className = 'inline-block px-4 py-1.5 rounded-full text-sm font-extrabold uppercase tracking-wider bg-amber-500/20 text-amber-400 border border-amber-500/30';
        } else {
            badgeElem.className = 'inline-block px-4 py-1.5 rounded-full text-sm font-extrabold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
        }
    } catch (err) {
        console.warn("Backend API not reachable. Using fallback heuristic simulation.");
        runFallbackSimulation(payload);
    }
}

function runFallbackSimulation(payload) {
    let score = (payload.density * 0.35) + (payload.gravity * 0.45) + (payload.purity * 0.02) + (payload.refractive * 0.5);
    score = Math.min(Math.max(score, 1.0), 9.8);

    document.getElementById('predicted-score').innerText = score.toFixed(2) + ' / 10';
    document.getElementById('m-hardness').innerText = score.toFixed(2);
    document.getElementById('m-recovery').innerText = (score * 9.5 + 10).toFixed(1) + '%';

    const badgeElem = document.getElementById('quality-badge');
    if (score < 3.5) {
        badgeElem.innerText = 'Low Quality';
        badgeElem.className = 'inline-block px-4 py-1.5 rounded-full text-sm font-extrabold uppercase tracking-wider bg-rose-500/20 text-rose-400 border border-rose-500/30';
    } else if (score <= 6.5) {
        badgeElem.innerText = 'Medium Quality';
        badgeElem.className = 'inline-block px-4 py-1.5 rounded-full text-sm font-extrabold uppercase tracking-wider bg-amber-500/20 text-amber-400 border border-amber-500/30';
    } else {
        badgeElem.innerText = 'High Quality';
        badgeElem.className = 'inline-block px-4 py-1.5 rounded-full text-sm font-extrabold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
    }
}

function renderCharts() {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#334155';

    new Chart(document.getElementById('chart-importance'), {
        type: 'bar',
        data: {
            labels: ['Specific Gravity', 'Calculated Density', 'Commodity Diversity', 'Refractive Index', 'Chemical Purity', 'Active Companies'],
            datasets: [{ data: [0.32, 0.27, 0.15, 0.12, 0.08, 0.06], backgroundColor: '#10b981', borderRadius: 6 }]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });

    new Chart(document.getElementById('chart-performance'), {
        type: 'bar',
        data: {
            labels: ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Multi-Source XGBoost'],
            datasets: [{ data: [0.65, 0.76, 0.89, 0.92, 0.96], backgroundColor: '#06b6d4', borderRadius: 6 }]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 1.0 } }, plugins: { legend: { display: false } } }
    });
}