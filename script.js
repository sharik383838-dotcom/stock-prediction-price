/**
 * AI Stock Predictor - Frontend JavaScript
 * Handles user interactions, API communication with Flask backend,
 * and renders interactive visualizations with Chart.js.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('prediction-form');
    const tickerInput = document.getElementById('ticker-input');
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    const loadingState = document.getElementById('loading-state');
    const loadingMessage = document.getElementById('loading-message');
    const errorBanner = document.getElementById('error-banner');
    const errorMessage = document.getElementById('error-message');
    const welcomeState = document.getElementById('welcome-state');
    const resultsDashboard = document.getElementById('results-dashboard');
    const symbolChips = document.querySelectorAll('.symbol-chip');

    // Result Display Elements
    const displayTicker = document.getElementById('display-ticker');
    const displayCurrency = document.getElementById('display-currency');
    const displayLatestDate = document.getElementById('display-latest-date');
    const displayTargetDate = document.getElementById('display-target-date');
    const trendBadge = document.getElementById('trend-indicator-badge');
    const trendIcon = document.getElementById('trend-icon');
    const trendText = document.getElementById('trend-text');

    const valLatestClose = document.getElementById('val-latest-close');
    const valLatestDateFooter = document.getElementById('val-latest-date-footer');
    const valPredictedClose = document.getElementById('val-predicted-close');
    const valPredictedDateFooter = document.getElementById('val-predicted-date-footer');
    const valPriceChange = document.getElementById('val-price-change');
    const valPriceChangePct = document.getElementById('val-price-change-pct');
    const changeIcon = document.getElementById('change-icon');
    const valMae = document.getElementById('val-mae');
    const valMaeFooter = document.getElementById('val-mae-footer');

    // Chart Instance Holder
    let stockChart = null;

    // Loading status rotation timer
    let loadingInterval = null;
    const loadingPhases = [
        "Fetching historical market data from Yahoo Finance...",
        "Engineering technical indicators & moving averages...",
        "Training Scikit-learn RandomForestRegressor...",
        "Validating out-of-sample chronological accuracy (MAE)...",
        "Generating next trading day forecast..."
    ];

    /**
     * Format currency display with appropriate symbol or code
     */
    function formatCurrency(amount, currencyCode = 'USD') {
        const num = Number(amount);
        if (isNaN(num)) return '$0.00';

        let symbol = '$';
        if (currencyCode === 'INR') symbol = '₹';
        else if (currencyCode === 'EUR') symbol = '€';
        else if (currencyCode === 'GBP') symbol = '£';
        else if (currencyCode === 'JPY') symbol = '¥';
        else if (currencyCode !== 'USD') symbol = `${currencyCode} `;

        return `${symbol}${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }

    /**
     * Input handlers
     */
    tickerInput.addEventListener('input', () => {
        tickerInput.value = tickerInput.value.toUpperCase();
        clearBtn.style.display = tickerInput.value.trim() ? 'block' : 'none';
    });

    clearBtn.addEventListener('click', () => {
        tickerInput.value = '';
        clearBtn.style.display = 'none';
        tickerInput.focus();
    });

    // Quick symbol suggestion chips
    symbolChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const symbol = chip.getAttribute('data-symbol');
            tickerInput.value = symbol;
            clearBtn.style.display = 'block';
            executePrediction(symbol);
        });
    });

    // Form submission
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const symbol = tickerInput.value.trim();
        if (symbol) {
            executePrediction(symbol);
        }
    });

    /**
     * Send prediction request to Flask backend
     */
    async function executePrediction(symbol) {
        showLoadingState();
        hideError();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ ticker: symbol })
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                const errorMsg = data.error || `Unable to fetch forecast for '${symbol}'. Please try again.`;
                showError(errorMsg);
                return;
            }

            renderDashboard(data);

        } catch (err) {
            console.error('Fetch error:', err);
            showError('Network or server error encountered. Please verify the Flask server is running.');
        } finally {
            hideLoadingState();
        }
    }

    /**
     * Update Dashboard DOM with ML forecast results
     */
    function renderDashboard(data) {
        const currency = data.currency || 'USD';
        const isBullish = data.trend === 'Bullish';

        // Header info
        displayTicker.textContent = data.ticker;
        displayCurrency.textContent = currency;
        displayLatestDate.textContent = data.latest_date_display;
        displayTargetDate.textContent = data.predicted_date_display;

        // Trend Badge
        if (isBullish) {
            trendBadge.className = 'trend-badge trend-bullish';
            trendIcon.textContent = '↗';
            trendText.textContent = 'BULLISH OUTLOOK';
        } else {
            trendBadge.className = 'trend-badge trend-bearish';
            trendIcon.textContent = '↘';
            trendText.textContent = 'BEARISH OUTLOOK';
        }

        // Latest Close Card
        valLatestClose.textContent = formatCurrency(data.latest_close, currency);
        valLatestDateFooter.textContent = `Actual close on ${data.latest_date_display}`;

        // Predicted Close Card (Hero)
        valPredictedClose.textContent = formatCurrency(data.predicted_price, currency);
        valPredictedDateFooter.textContent = `Target: ${data.predicted_date_display}`;

        // Expected Price Change Card
        const sign = data.price_change >= 0 ? '+' : '';
        valPriceChange.textContent = `${sign}${formatCurrency(data.price_change, currency)}`;
        valPriceChangePct.textContent = `${sign}${data.price_change_pct.toFixed(2)}% vs latest close`;

        if (isBullish) {
            valPriceChange.className = 'metric-value color-bullish';
            valPriceChangePct.className = 'metric-footer color-bullish';
            changeIcon.style.color = 'var(--bullish-green)';
            changeIcon.innerHTML = `
                <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none">
                    <line x1="12" y1="19" x2="12" y2="5"></line>
                    <polyline points="5 12 12 5 19 12"></polyline>
                </svg>`;
        } else {
            valPriceChange.className = 'metric-value color-bearish';
            valPriceChangePct.className = 'metric-footer color-bearish';
            changeIcon.style.color = 'var(--bearish-red)';
            changeIcon.innerHTML = `
                <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <polyline points="19 12 12 19 5 12"></polyline>
                </svg>`;
        }

        // Model Evaluation (MAE)
        valMae.textContent = formatCurrency(data.mae, currency);
        valMaeFooter.textContent = `Test set error: ±${data.mae_pct.toFixed(2)}% (${data.test_samples} unseen trading days)`;

        // Render Chart
        renderStockChart(data);

        // Show results dashboard and hide welcome card
        if (welcomeState) welcomeState.style.display = 'none';
        resultsDashboard.style.display = 'flex';

        // Smooth scroll to results
        resultsDashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    /**
     * Render Interactive Stock Chart using Chart.js
     * Displays actual historical prices with a distinct projected point for tomorrow
     */
    function renderStockChart(data) {
        const ctx = document.getElementById('stockChart').getContext('2d');

        // Extract historical dates and closes
        const histData = data.historical_data || [];
        const labels = histData.map(item => item.displayDate);
        const actualPrices = histData.map(item => item.close);

        // Add next trading day to labels
        const nextDateLabel = `${data.predicted_date_display.split(',')[0]} (Forecast)`;
        labels.push(nextDateLabel);

        // Actual dataset has null for the forecast slot
        const actualSeries = [...actualPrices, null];

        // Forecast projection series:
        // Connects from the last known actual price to the predicted price
        const forecastSeries = new Array(actualPrices.length - 1).fill(null);
        forecastSeries.push(actualPrices[actualPrices.length - 1]); // Last actual close
        forecastSeries.push(data.predicted_price); // Predicted close

        // Point styling arrays for projection series
        const pointRadii = new Array(actualPrices.length).fill(0);
        pointRadii.push(7); // Highlighted predicted point

        const pointHoverRadii = new Array(actualPrices.length).fill(0);
        pointHoverRadii.push(11);

        // Gradient for historical fill
        const gradient = ctx.createLinearGradient(0, 0, 0, 350);
        gradient.addColorStop(0, 'rgba(168, 85, 247, 0.35)');
        gradient.addColorStop(1, 'rgba(168, 85, 247, 0.00)');

        // Destroy previous chart if exists
        if (stockChart) {
            stockChart.destroy();
        }

        stockChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Historical Close',
                        data: actualSeries,
                        borderColor: '#a855f7',
                        borderWidth: 2.2,
                        backgroundColor: gradient,
                        fill: true,
                        tension: 0.25,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        pointBackgroundColor: '#a855f7',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 1.5,
                        order: 2
                    },
                    {
                        label: 'Predicted Next Close',
                        data: forecastSeries,
                        borderColor: '#ec4899',
                        borderWidth: 2.5,
                        borderDash: [6, 6],
                        backgroundColor: 'transparent',
                        fill: false,
                        tension: 0,
                        pointRadius: pointRadii,
                        pointHoverRadius: pointHoverRadii,
                        pointBackgroundColor: '#ec4899',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        order: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(18, 13, 36, 0.95)',
                        titleColor: '#f8fafc',
                        titleFont: {
                            family: 'Outfit',
                            size: 13,
                            weight: '600'
                        },
                        bodyColor: '#cbd5e1',
                        bodyFont: {
                            family: 'Inter',
                            size: 12
                        },
                        borderColor: 'rgba(168, 85, 247, 0.4)',
                        borderWidth: 1,
                        padding: 12,
                        boxPadding: 6,
                        usePointStyle: true,
                        callbacks: {
                            label: function(context) {
                                const datasetLabel = context.dataset.label || '';
                                const value = context.parsed.y;
                                if (value === null || value === undefined) return null;

                                if (context.datasetIndex === 1 && context.dataIndex === labels.length - 1) {
                                    return `🔮 ${datasetLabel}: ${formatCurrency(value, data.currency)} (AI Estimate)`;
                                } else if (context.datasetIndex === 0) {
                                    return `📈 Actual Close: ${formatCurrency(value, data.currency)}`;
                                }
                                return null;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(168, 85, 247, 0.07)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#94a3b8',
                            font: {
                                family: 'Inter',
                                size: 11
                            },
                            maxTicksLimit: 10,
                            maxRotation: 0
                        }
                    },
                    y: {
                        position: 'right',
                        grid: {
                            color: 'rgba(168, 85, 247, 0.08)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#94a3b8',
                            font: {
                                family: 'Inter',
                                size: 11
                            },
                            callback: function(value) {
                                return formatCurrency(value, data.currency);
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * UI State Helpers
     */
    function showLoadingState() {
        predictBtn.disabled = true;
        predictBtn.classList.add('is-loading');
        tickerInput.disabled = true;
        loadingState.style.display = 'flex';

        let phaseIndex = 0;
        loadingMessage.textContent = loadingPhases[0];
        loadingInterval = setInterval(() => {
            phaseIndex = (phaseIndex + 1) % loadingPhases.length;
            loadingMessage.textContent = loadingPhases[phaseIndex];
        }, 1200);
    }

    function hideLoadingState() {
        predictBtn.disabled = false;
        predictBtn.classList.remove('is-loading');
        tickerInput.disabled = false;
        loadingState.style.display = 'none';
        if (loadingInterval) {
            clearInterval(loadingInterval);
            loadingInterval = null;
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.style.display = 'flex';
        errorBanner.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideError() {
        errorBanner.style.display = 'none';
        errorMessage.textContent = '';
    }
});
