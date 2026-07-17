// script.js
document.addEventListener('DOMContentLoaded', function() {

  // ─── QR Code Generation ──────────────────────────────────────────
  function generateQR(containerId, text) {
    const container = document.getElementById(containerId);
    if (!container) return;
    // Clear any existing QR (in case of regeneration)
    container.innerHTML = '';
    // Add a label (optional – we keep the label already in HTML)
    // but we need to keep the label; we'll re-insert it after QR
    try {
      new QRCode(container, {
        text: text,
        width: 140,    // in pixels, will be sized via CSS
        height: 140,
        colorDark: '#1a0d02',
        colorLight: '#ffffff',
        correctLevel: QRCode.CorrectLevel.H
      });
      // Restore the label if needed (the QRCode library may clear the container)
      // We'll re-append the label
      const label = document.createElement('div');
      label.className = 'qr-label';
      label.textContent = 'Scan';
      container.appendChild(label);
    } catch (e) {
      console.error('QR generation failed for ' + containerId, e);
    }
  }

  // Generate both QR codes with sample tokens
  generateQR('qrCandidate', 'VXK8P2M47QFN');
  generateQR('qrAgenda', 'AGN37X1F8Q22');

  // ─── Bubble Toggle Logic ─────────────────────────────────────────
  // We use event delegation on each card
  document.querySelectorAll('.voter-card').forEach(card => {
    card.addEventListener('click', function(e) {
      // Find the clicked bubble element
      let bubble = e.target.closest('.omr-bubble');
      if (!bubble) {
        // If user clicked on the row item, find the bubble inside it
        const rowItem = e.target.closest('.candidate-item, .agenda-option');
        if (rowItem) {
          bubble = rowItem.querySelector('.omr-bubble');
        }
      }
      if (!bubble) return;

      // Toggle the 'checked' class
      bubble.classList.toggle('checked');

      // Optionally, enforce "only one" in candidate card (single selection)
      const card = bubble.closest('.voter-card');
      if (card && card.id === 'cardCandidate') {
        // If the bubble is now checked, uncheck all other bubbles in this card
        if (bubble.classList.contains('checked')) {
          const allBubbles = card.querySelectorAll('.omr-bubble');
          allBubbles.forEach(b => {
            if (b !== bubble) b.classList.remove('checked');
          });
        }
      }
      // For agenda cards, we allow multiple selections (one per row) – no extra logic needed.
    });
  });

  // ─── Optional: Reset button (if you add one) ──────────────────
  // You can add a button in the HTML to clear all bubbles.
  // For now we omit it.
});
