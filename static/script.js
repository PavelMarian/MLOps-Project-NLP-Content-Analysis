function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('ru-RU');
}

function formatPercent(value) {
    return (value * 100).toFixed(1) + '%';
}

function getBadgeClass(isToxic, confidence) {
    if (!isToxic) return 'badge-safe';
    if (confidence > 0.8) return 'badge-toxic';
    return 'badge-warning';
}

function getBadgeText(isToxic, confidence) {
    if (!isToxic) return '✅ Безопасно';
    if (confidence > 0.8) return '🚫 Токсично';
    return '⚠️ Подозрительно';
}

window.showNotification = function(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
        <span>${type === 'warning' ? '⚠️' : type === 'error' ? '❌' : 'ℹ️'}</span>
        <span>${message}</span>
        <button class="close-btn" onclick="this.parentElement.remove()">×</button>
    `;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
};
