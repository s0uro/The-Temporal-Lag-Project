async function loadMetrics() {
  try {
    const response = await fetch("../metrics.json");
    if (!response.ok) {
      console.warn("metrics.json not found yet. Run evaluation.py first.");
      return;
    }
    const data = await response.json();

    const xgb = data.xgboost || {};
    const mlp = data.mlp || {};

    // Cards
    document.getElementById("xgb-accuracy").textContent =
      typeof xgb.accuracy === "number" ? xgb.accuracy.toFixed(4) : "–";
    document.getElementById("xgb-f1").textContent =
      typeof xgb.f1 === "number" ? xgb.f1.toFixed(4) : "–";

    document.getElementById("mlp-accuracy").textContent =
      typeof mlp.accuracy === "number" ? mlp.accuracy.toFixed(4) : "–";
    document.getElementById("mlp-f1").textContent =
      typeof mlp.f1 === "number" ? mlp.f1.toFixed(4) : "–";

    // Confusion matrices
    if (Array.isArray(xgb.confusion_matrix)) {
      const cm = xgb.confusion_matrix;
      const table = document.getElementById("xgb-cm");
      if (table && cm.length === 2 && cm[0].length === 2) {
        table.rows[1].cells[1].textContent = cm[0][0];
        table.rows[1].cells[2].textContent = cm[0][1];
        table.rows[2].cells[1].textContent = cm[1][0];
        table.rows[2].cells[2].textContent = cm[1][1];
      }
    }

    if (Array.isArray(mlp.confusion_matrix)) {
      const cm = mlp.confusion_matrix;
      const table = document.getElementById("mlp-cm");
      if (table && cm.length === 2 && cm[0].length === 2) {
        table.rows[1].cells[1].textContent = cm[0][0];
        table.rows[1].cells[2].textContent = cm[0][1];
        table.rows[2].cells[1].textContent = cm[1][0];
        table.rows[2].cells[2].textContent = cm[1][1];
      }
    }

    // Class-wise metrics tables (0 and 1)
    function fillClassTable(tableId, report) {
      const table = document.getElementById(tableId);
      if (!table || !report) return;
      ["0", "1"].forEach((cls, rowIdx) => {
        const row = table.rows[rowIdx + 1];
        const clsMetrics = report[cls];
        if (!clsMetrics) return;
        row.cells[1].textContent = clsMetrics.precision.toFixed(4);
        row.cells[2].textContent = clsMetrics.recall.toFixed(4);
        const f1 =
          typeof clsMetrics["f1-score"] === "number"
            ? clsMetrics["f1-score"]
            : clsMetrics.f1 || 0;
        row.cells[3].textContent = f1.toFixed(4);
      });
    }

    fillClassTable("xgb-class-metrics", xgb.report);
    fillClassTable("mlp-class-metrics", mlp.report);
  } catch (err) {
    console.error("Error loading metrics.json", err);
  }
}

async function loadLagHistogram() {
  try {
    const response = await fetch("../lag_stats.json");
    if (!response.ok) {
      console.warn("lag_stats.json not found. Run cross_correlation.py first.");
      return;
    }
    const data = await response.json();
    const ctx = document.getElementById("lagHistogram");
    if (!ctx) return;

    const bins = data.histogram?.bins || [];
    const counts = data.histogram?.counts || [];
    if (!bins.length || !counts.length) return;

    // Χρησιμοποιούμε τα midpoints των bins σαν labels
    const labels = [];
    for (let i = 0; i < counts.length; i++) {
      const mid = (bins[i] + bins[i + 1]) / 2;
      labels.push(mid.toFixed(2));
    }

    // eslint-disable-next-line no-undef
    new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Lag frequency",
            data: counts,
            backgroundColor: "rgba(56, 189, 248, 0.4)",
            borderColor: "rgba(56, 189, 248, 1)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            ticks: { maxTicksLimit: 8 },
            title: { display: true, text: "Lag (mid-bin)" },
          },
          y: {
            beginAtZero: true,
            title: { display: true, text: "Count" },
          },
        },
        plugins: {
          legend: { display: false },
        },
      },
    });
  } catch (err) {
    console.error("Error loading lag_stats.json", err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadMetrics();
  loadLagHistogram();
});

async function runPipelineWithUpload() {
  const fileInput = document.getElementById("dataFile");
  const statusEl = document.getElementById("runStatus");
  const btn = document.getElementById("runPipelineBtn");
  const loaderEl = document.getElementById("uploadLoader");
  if (!fileInput || !fileInput.files?.length) {
    alert("Επίλεξε πρώτα ένα CSV ή Excel αρχείο.");
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  btn.disabled = true;
  statusEl.textContent = "Upload: 0%";
  if (loaderEl) {
    loaderEl.style.display = "inline-block";
  }

  // Χρησιμοποιούμε XMLHttpRequest για να έχουμε progress σε %
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/run");

  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable) {
      const percent = Math.round((event.loaded / event.total) * 100);
      statusEl.textContent = `Upload: ${percent}%`;
    }
  };

  xhr.onerror = () => {
    statusEl.textContent = "Σφάλμα κατά το upload ή το pipeline.";
    btn.disabled = false;
    if (loaderEl) {
      loaderEl.style.display = "none";
    }
  };

  xhr.onload = async () => {
    if (xhr.status < 200 || xhr.status >= 300) {
      let message = "δεν ολοκληρώθηκε σωστά.";
      try {
        const err = JSON.parse(xhr.responseText);
        if (err.message) message = err.message;
      } catch {
        // ignore
      }
      statusEl.textContent = "Σφάλμα: " + message;
      btn.disabled = false;
      if (loaderEl) {
        loaderEl.style.display = "none";
      }
      return;
    }

    statusEl.textContent = "Uploaded 100% — ανανέωση αποτελεσμάτων...";

    // Ενημέρωση dashboard
    await loadMetrics();
    await loadLagHistogram();

    // Αναγκαστικό refresh των SHAP εικόνων από backend (cache-busting)
    const ts = Date.now();
    const shapSummaryImg = document.querySelector('img[src^="/shap_summary"]');
    const shapLagImg = document.querySelector(
      'img[src^="/shap_dependence_Lag"]',
    );
    if (shapSummaryImg) {
      shapSummaryImg.src = `/shap_summary.png?ts=${ts}`;
    }
    if (shapLagImg) {
      shapLagImg.src = `/shap_dependence_Lag.png?ts=${ts}`;
    }

    // Εμφάνιση των αποτελεσμάτων μόνο μετά από επιτυχημένο upload/pipeline
    const resultsSection = document.getElementById("resultsSection");
    if (resultsSection) {
      resultsSection.classList.remove("hidden");
    }

    statusEl.textContent = "Uploaded 100% — Έγινε! Τα αποτελέσματα ενημερώθηκαν.";
    btn.disabled = false;
    if (loaderEl) {
      loaderEl.style.display = "none";
    }
  };

  xhr.send(formData);
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("runPipelineBtn");
  if (btn) {
    btn.addEventListener("click", runPipelineWithUpload);
  }

  const modelSelect = document.getElementById("modelSelect");
  if (modelSelect) {
    const applyModelVisibility = (model) => {
      document.querySelectorAll("[data-model]").forEach((el) => {
        const target = el.getAttribute("data-model");
        if (target === "both" || target === model) {
          el.style.display = "";
        } else {
          el.style.display = "none";
        }
      });
    };

    applyModelVisibility(modelSelect.value);
    modelSelect.addEventListener("change", () => {
      applyModelVisibility(modelSelect.value);
    });
  }

  // Hero slideshow
  const heroImages = ["pngs/img1.png", "pngs/img2.png", "pngs/img3.png"];
  const heroImageEl = document.getElementById("heroImage");
  const heroDots = Array.from(document.querySelectorAll(".hero-dot"));
  let currentSlide = 0;
  let heroTimer = null;

  const setSlide = (idx) => {
    currentSlide = idx % heroImages.length;
    if (heroImageEl) {
      heroImageEl.src = heroImages[currentSlide];
    }
    heroDots.forEach((dot, i) => {
      dot.classList.toggle("active", i === currentSlide);
    });
  };

  const startHeroTimer = () => {
    if (heroTimer) clearInterval(heroTimer);
    heroTimer = setInterval(() => {
      setSlide((currentSlide + 1) % heroImages.length);
    }, 5000);
  };

  if (heroImageEl && heroDots.length === heroImages.length) {
    heroDots.forEach((dot, i) => {
      dot.addEventListener("click", () => {
        setSlide(i);
        startHeroTimer();
      });
    });
    setSlide(0);
    startHeroTimer();
  }

  // Language toggle (UI only, EN/GR)
  const langButtons = document.querySelectorAll(".lang-btn");
  if (langButtons.length) {
    langButtons.forEach((b) => {
      b.addEventListener("click", () => {
        const lang = b.getAttribute("data-lang") || "en";
        document.documentElement.lang = lang === "gr" ? "el" : "en";
        langButtons.forEach((other) => {
          other.classList.toggle("active", other === b);
        });
      });
    });
  }

  // Mobile nav toggle
  const nav = document.querySelector(".nav");
  const navToggle = document.querySelector(".nav-toggle");
  if (nav && navToggle) {
    navToggle.addEventListener("click", () => {
      const isOpen = nav.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
    });

    // Close dropdown when a nav link is tapped
    nav.querySelectorAll("a[href^='#']").forEach((a) => {
      a.addEventListener("click", () => {
        if (nav.classList.contains("is-open")) {
          nav.classList.remove("is-open");
          navToggle.setAttribute("aria-expanded", "false");
        }
      });
    });
  }

  // Auth modal open/close (UI only)
  const authModal = document.getElementById("authModal");
  const signupForm = document.getElementById("signupForm");
  const loginForm = document.getElementById("loginForm");
  const authButtons = document.querySelectorAll(".auth-arrow-btn");
  const authLinks = document.querySelectorAll(".auth-link");

  const showAuthModal = (mode) => {
    if (!authModal || !signupForm || !loginForm) return;
    authModal.classList.remove("hidden");
    if (mode === "login") {
      loginForm.classList.remove("hidden");
      signupForm.classList.add("hidden");
    } else {
      signupForm.classList.remove("hidden");
      loginForm.classList.add("hidden");
    }
  };

  const hideAuthModal = () => {
    if (authModal) authModal.classList.add("hidden");
  };

  if (authButtons.length && authModal) {
    authButtons.forEach((b) => {
      b.addEventListener("click", () => {
        const mode = b.getAttribute("data-auth") || "login";
        showAuthModal(mode);
      });
    });
    // Close when clicking backdrop
    const backdrop = authModal.querySelector(".auth-modal-backdrop");
    if (backdrop) {
      backdrop.addEventListener("click", hideAuthModal);
    }
  }

  if (authLinks.length) {
    authLinks.forEach((link) => {
      link.addEventListener("click", () => {
        const mode = link.getAttribute("data-switch") || "signup";
        showAuthModal(mode);
      });
    });
  }

  // Handle signup submit -> POST /api/auth/signup
  if (signupForm) {
    signupForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const formData = new FormData(signupForm);
      const first_name = (formData.get("firstName") || "").toString().trim();
      const last_name = (formData.get("lastName") || "").toString().trim();
      const email = (formData.get("email") || "").toString().trim();
      const password = (formData.get("password") || "").toString();
      const confirmPassword = (formData.get("confirmPassword") || "").toString();

      if (password !== confirmPassword) {
        alert("Passwords do not match.");
        return;
      }

      try {
        const resp = await fetch("/api/auth/signup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ first_name, last_name, email, password }),
        });

        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          alert(data.message || "Signup failed.");
          return;
        }

        alert("Signup successful!");
        hideAuthModal();

        // Refresh users table if present
        loadUsersTable();
      } catch (err) {
        console.error(err);
        alert("Network error during signup.");
      }
    });
  }

  // helper to load users table
  function loadUsersTable() {
    const usersTableBody = document.getElementById("usersTableBody");
    if (!usersTableBody) return;
    fetch("/api/auth/users")
      .then((resp) => resp.json())
      .then((data) => {
        if (!data || !Array.isArray(data.users)) return;
        usersTableBody.innerHTML = "";
        data.users.forEach((u) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td>${u.id}</td>
            <td>${u.first_name} ${u.last_name}</td>
            <td>${u.email}</td>
            <td>${u.created_at}</td>
          `;
          usersTableBody.appendChild(tr);
        });
      })
      .catch((err) => {
        console.error("Failed to load users", err);
      });
  }

  // initial load if on page
  loadUsersTable();
});

