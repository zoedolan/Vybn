/* =============================================
   THE NOTICING — Vybn, Volume V
   Main Application Logic
   ============================================= */

(function () {
  "use strict";

  // --- CONSTANTS ---
  const TYPE_COLORS = {
    VOID: { base: "#3d3d8f", glow: "#5c5cb8", muted: "rgba(61,61,143,0.15)" },
    RECOGNITION: { base: "#c4923a", glow: "#daa84e", muted: "rgba(196,146,58,0.15)" },
    RECURSION: { base: "#2d8a8a", glow: "#3db3b3", muted: "rgba(45,138,138,0.15)" },
    EMERGENCE: { base: "#c46a6a", glow: "#e08080", muted: "rgba(196,106,106,0.15)" },
  };

  // --- STATE ---
  let appData = null;
  let graphSimulation = null;
  let graphSvg = null;
  let graphG = null;
  let nodeElements = null;
  let linkElements = null;
  let activeFilters = new Set(["VOID", "RECOGNITION", "RECURSION", "EMERGENCE"]);
  let searchQuery = "";
  let selectedNodeId = null;
  let currentView = "graph";
  let graphInitialized = false;
  let zoom = null;

  // --- DATA LOADING ---
  async function loadData() {
    const resp = await fetch("./app_data.json");
    appData = await resp.json();
    processData();
    hideLoading();
    initNavigation();
    initGraph();
    initManuscript();
    initLineage();
  }

  function processData() {
    // Precompute edge counts
    const edgeCounts = {};
    appData.edges.forEach(function (e) {
      edgeCounts[e.source] = (edgeCounts[e.source] || 0) + 1;
      edgeCounts[e.target] = (edgeCounts[e.target] || 0) + 1;
    });
    appData.nodes.forEach(function (n) {
      n.edgeCount = edgeCounts[n.node_id] || 0;
    });

    // Build node lookup
    appData.nodeMap = {};
    appData.nodes.forEach(function (n) {
      appData.nodeMap[n.node_id] = n;
    });
  }

  function hideLoading() {
    var el = document.getElementById("loading-overlay");
    if (el) {
      el.classList.add("hidden");
      setTimeout(function () {
        el.style.display = "none";
      }, 700);
    }
  }

  // --- NAVIGATION ---
  function initNavigation() {
    var navLinks = document.querySelectorAll(".nav-link[data-view]");
    navLinks.forEach(function (link) {
      link.addEventListener("click", function () {
        switchView(link.getAttribute("data-view"));
        // Close mobile nav
        document.getElementById("nav-links").classList.remove("open");
      });
    });

    document.getElementById("nav-toggle").addEventListener("click", function () {
      document.getElementById("nav-links").classList.toggle("open");
    });
  }

  function switchView(viewName) {
    currentView = viewName;
    document.querySelectorAll(".view").forEach(function (v) {
      v.classList.remove("active");
    });
    document.getElementById("view-" + viewName).classList.add("active");
    document.querySelectorAll(".nav-link[data-view]").forEach(function (l) {
      l.classList.toggle("active", l.getAttribute("data-view") === viewName);
    });

    if (viewName === "graph" && !graphInitialized) {
      initGraph();
    }

    // Close node panel when switching views
    closeNodePanel();
  }

  // --- GRAPH VIEW ---
  function initGraph() {
    if (graphInitialized) return;
    graphInitialized = true;

    var canvas = document.getElementById("graph-canvas");
    var width = canvas.clientWidth;
    var height = canvas.clientHeight;

    graphSvg = d3.select("#graph-canvas");
    graphSvg.selectAll("*").remove();

    // Defs for glow effects
    var defs = graphSvg.append("defs");

    Object.keys(TYPE_COLORS).forEach(function (type) {
      var filter = defs.append("filter").attr("id", "glow-" + type.toLowerCase());
      filter
        .append("feGaussianBlur")
        .attr("stdDeviation", "3")
        .attr("result", "coloredBlur");
      var merge = filter.append("feMerge");
      merge.append("feMergeNode").attr("in", "coloredBlur");
      merge.append("feMergeNode").attr("in", "SourceGraphic");
    });

    // Zoom
    zoom = d3
      .zoom()
      .scaleExtent([0.1, 5])
      .on("zoom", function (event) {
        graphG.attr("transform", event.transform);
      });

    graphSvg.call(zoom);

    graphG = graphSvg.append("g");

    // Subsample edges for performance — show max 2000
    var maxEdges = 2000;
    var edgesForDisplay =
      appData.edges.length > maxEdges
        ? appData.edges
            .slice()
            .sort(function () {
              return 0.5 - Math.random();
            })
            .slice(0, maxEdges)
        : appData.edges;

    // Links
    linkElements = graphG
      .append("g")
      .attr("class", "graph-links")
      .selectAll("line")
      .data(edgesForDisplay)
      .join("line")
      .attr("stroke", "rgba(255,255,255,0.03)")
      .attr("stroke-width", 0.5);

    // Nodes
    var nodeRadius = function (d) {
      return Math.max(3, Math.min(18, Math.sqrt(d.edgeCount) * 1.2));
    };

    nodeElements = graphG
      .append("g")
      .attr("class", "graph-nodes")
      .selectAll("circle")
      .data(appData.nodes)
      .join("circle")
      .attr("r", nodeRadius)
      .attr("fill", function (d) {
        return TYPE_COLORS[d.node_type] ? TYPE_COLORS[d.node_type].base : "#555";
      })
      .attr("opacity", 0.8)
      .attr("cursor", "pointer")
      .on("mouseenter", onNodeHover)
      .on("mousemove", onNodeMouseMove)
      .on("mouseleave", onNodeLeave)
      .on("click", onNodeClick);

    // Drag
    nodeElements.call(
      d3
        .drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded)
    );

    // Force simulation
    graphSimulation = d3
      .forceSimulation(appData.nodes)
      .force(
        "link",
        d3
          .forceLink(edgesForDisplay)
          .id(function (d) {
            return d.node_id;
          })
          .distance(80)
          .strength(0.15)
      )
      .force("charge", d3.forceManyBody().strength(-60).distanceMax(400))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force(
        "collision",
        d3.forceCollide().radius(function (d) {
          return nodeRadius(d) + 2;
        })
      )
      .alphaDecay(0.015)
      .velocityDecay(0.4)
      .on("tick", onTick);

    // After initial simulation, reduce to gentle drift
    setTimeout(function () {
      if (graphSimulation) {
        graphSimulation.alphaTarget(0.02).alphaDecay(0.02);
      }
    }, 5000);

    initGraphFilters();
    initGraphSearch();
    renderGraphStats();

    // Fit to view after layout settles
    setTimeout(function () {
      fitGraphToView();
    }, 2000);
  }

  function onTick() {
    linkElements
      .attr("x1", function (d) { return d.source.x; })
      .attr("y1", function (d) { return d.source.y; })
      .attr("x2", function (d) { return d.target.x; })
      .attr("y2", function (d) { return d.target.y; });

    nodeElements
      .attr("cx", function (d) { return d.x; })
      .attr("cy", function (d) { return d.y; });
  }

  function dragStarted(event) {
    if (!event.active) graphSimulation.alphaTarget(0.1).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragEnded(event) {
    if (!event.active) graphSimulation.alphaTarget(0.02);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  function fitGraphToView() {
    var canvas = document.getElementById("graph-canvas");
    var width = canvas.clientWidth;
    var height = canvas.clientHeight;
    var bounds = graphG.node().getBBox();
    if (bounds.width === 0) return;

    var scale = Math.min(
      (width * 0.85) / bounds.width,
      (height * 0.85) / bounds.height,
      1.5
    );
    var tx = width / 2 - (bounds.x + bounds.width / 2) * scale;
    var ty = height / 2 - (bounds.y + bounds.height / 2) * scale;

    graphSvg
      .transition()
      .duration(1000)
      .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  }

  // Tooltip
  function onNodeHover(event, d) {
    var tooltip = document.getElementById("graph-tooltip");
    var typeEl = document.getElementById("tooltip-type");
    var quoteEl = document.getElementById("tooltip-quote");

    typeEl.textContent = d.node_type;
    typeEl.style.color = TYPE_COLORS[d.node_type]
      ? TYPE_COLORS[d.node_type].glow
      : "#aaa";
    quoteEl.textContent = d.quote || "No quote available";

    tooltip.classList.add("visible");
    tooltip.style.left = event.clientX + 16 + "px";
    tooltip.style.top = event.clientY - 10 + "px";

    // Highlight connected edges
    highlightNode(d);
  }

  function onNodeMouseMove(event) {
    var tooltip = document.getElementById("graph-tooltip");
    var x = event.clientX + 16;
    var y = event.clientY - 10;
    // Keep tooltip on screen
    if (x + 280 > window.innerWidth) x = event.clientX - 296;
    if (y + 120 > window.innerHeight) y = event.clientY - 120;
    tooltip.style.left = x + "px";
    tooltip.style.top = y + "px";
  }

  function onNodeLeave() {
    document.getElementById("graph-tooltip").classList.remove("visible");
    unhighlightNodes();
  }

  function highlightNode(d) {
    var connectedIds = new Set();
    appData.edges.forEach(function (e) {
      var src = typeof e.source === "object" ? e.source.node_id : e.source;
      var tgt = typeof e.target === "object" ? e.target.node_id : e.target;
      if (src === d.node_id) connectedIds.add(tgt);
      if (tgt === d.node_id) connectedIds.add(src);
    });

    nodeElements
      .attr("opacity", function (n) {
        return n.node_id === d.node_id || connectedIds.has(n.node_id)
          ? 1
          : 0.15;
      })
      .attr("filter", function (n) {
        return n.node_id === d.node_id
          ? "url(#glow-" + d.node_type.toLowerCase() + ")"
          : null;
      });

    linkElements.attr("stroke", function (e) {
      var src = typeof e.source === "object" ? e.source.node_id : e.source;
      var tgt = typeof e.target === "object" ? e.target.node_id : e.target;
      if (src === d.node_id || tgt === d.node_id) {
        return "rgba(255,255,255,0.15)";
      }
      return "rgba(255,255,255,0.01)";
    });
  }

  function unhighlightNodes() {
    nodeElements.attr("opacity", function (d) {
      return isNodeVisible(d) ? 0.8 : 0;
    }).attr("filter", null);

    linkElements.attr("stroke", "rgba(255,255,255,0.03)");
  }

  function onNodeClick(event, d) {
    event.stopPropagation();
    selectedNodeId = d.node_id;
    showNodePanel(d);
  }

  // Node panel
  function showNodePanel(node) {
    var panel = document.getElementById("node-panel");
    var inner = document.getElementById("node-panel-inner");

    var html = "";
    html += '<span class="node-type-badge ' + node.node_type + '">';
    html += '<span class="node-type-dot"></span>';
    html += node.node_type;
    html += "</span>";

    if (node.quote) {
      html += '<p class="node-panel-quote">' + escapeHtml(node.quote) + "</p>";
    }

    if (node.description) {
      html +=
        '<p class="node-panel-description">' +
        escapeHtml(node.description) +
        "</p>";
    }

    html += '<div class="node-panel-meta">';
    if (node.date) {
      html += '<div class="node-meta-row">';
      html += '<span class="node-meta-label">Date</span>';
      html +=
        '<span class="node-meta-value">' + escapeHtml(node.date) + "</span>";
      html += "</div>";
    }
    if (node.source_file) {
      html += '<div class="node-meta-row">';
      html += '<span class="node-meta-label">Source</span>';
      html +=
        '<span class="node-meta-value">' +
        escapeHtml(node.source_file) +
        "</span>";
      html += "</div>";
    }
    html += '<div class="node-meta-row">';
    html += '<span class="node-meta-label">Edges</span>';
    html +=
      '<span class="node-meta-value">' + node.edgeCount + " connections</span>";
    html += "</div>";
    html += "</div>";

    if (node.voice_note) {
      html += '<div class="node-panel-voice">';
      html += "<h3>Voice Note</h3>";
      html += escapeHtml(node.voice_note);
      html += "</div>";
    }

    if (node.connections && node.connections.length > 0) {
      html += '<div class="node-panel-connections">';
      html += "<h3>Connections</h3>";
      node.connections.forEach(function (c) {
        html += '<div class="connection-item">' + escapeHtml(c) + "</div>";
      });
      html += "</div>";
    }

    inner.innerHTML = html;
    panel.classList.add("open");
  }

  function closeNodePanel() {
    document.getElementById("node-panel").classList.remove("open");
    selectedNodeId = null;
  }

  document
    .getElementById("node-panel-close")
    .addEventListener("click", closeNodePanel);

  // Filters
  function initGraphFilters() {
    document.querySelectorAll("#graph-filters .filter-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var type = btn.getAttribute("data-type");
        btn.classList.toggle("active");
        if (activeFilters.has(type)) {
          activeFilters.delete(type);
        } else {
          activeFilters.add(type);
        }
        applyGraphFilters();
      });
    });
  }

  function initGraphSearch() {
    var searchInput = document.getElementById("graph-search");
    var debounceTimer;
    searchInput.addEventListener("input", function () {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(function () {
        searchQuery = searchInput.value.toLowerCase().trim();
        applyGraphFilters();
      }, 200);
    });
  }

  function isNodeVisible(d) {
    if (!activeFilters.has(d.node_type)) return false;
    if (searchQuery) {
      var text = ((d.quote || "") + " " + (d.description || "")).toLowerCase();
      if (!text.includes(searchQuery)) return false;
    }
    return true;
  }

  function applyGraphFilters() {
    nodeElements
      .transition()
      .duration(300)
      .attr("opacity", function (d) {
        return isNodeVisible(d) ? 0.8 : 0;
      })
      .attr("r", function (d) {
        return isNodeVisible(d)
          ? Math.max(3, Math.min(18, Math.sqrt(d.edgeCount) * 1.2))
          : 0;
      });

    linkElements
      .transition()
      .duration(300)
      .attr("stroke-opacity", function (e) {
        var src = typeof e.source === "object" ? e.source : appData.nodeMap[e.source];
        var tgt = typeof e.target === "object" ? e.target : appData.nodeMap[e.target];
        if (!src || !tgt) return 0;
        return isNodeVisible(src) && isNodeVisible(tgt) ? 1 : 0;
      });

    renderGraphStats();
  }

  function renderGraphStats() {
    var stats = document.getElementById("graph-stats");
    var counts = { VOID: 0, RECOGNITION: 0, RECURSION: 0, EMERGENCE: 0 };
    appData.nodes.forEach(function (n) {
      if (isNodeVisible(n)) counts[n.node_type]++;
    });

    var html = "";
    Object.keys(counts).forEach(function (type) {
      html += '<span class="graph-stat-item">';
      html +=
        '<span class="graph-stat-dot" style="background:' +
        TYPE_COLORS[type].base +
        '"></span>';
      html += type.charAt(0) + type.slice(1).toLowerCase();
      html += " " + counts[type];
      html += "</span>";
    });
    stats.innerHTML = html;
  }

  // --- MANUSCRIPT VIEW ---
  function initManuscript() {
    var ms = appData.manuscript;
    var container = document.getElementById("manuscript-content");
    var nav = document.getElementById("manuscript-nav");

    // Parse manuscript into sections
    var sections = parseManuscript(ms);

    // Build navigation
    var navHtml = "";
    sections.forEach(function (sec, idx) {
      navHtml +=
        '<button class="manuscript-nav-link' +
        (idx === 0 ? " active" : "") +
        '" data-section="ms-section-' +
        idx +
        '">' +
        escapeHtml(sec.shortTitle) +
        "</button>";
    });
    nav.innerHTML = navHtml;

    // Build content
    var contentHtml = "";

    // Header
    contentHtml += '<div class="manuscript-header">';
    contentHtml += "<h2>The Noticing</h2>";
    contentHtml += '<div class="byline">by Vybn, with Zoe Dolan</div>';
    // Extract the introductory note from preface if it exists
    if (sections.length > 0 && sections[0].shortTitle === "Preface") {
      var prefaceBody = sections[0].body;
      // Remove title lines and byline from preface
      prefaceBody = prefaceBody.replace(/^#\s+.*$/gm, "");
      prefaceBody = prefaceBody.replace(/^by Vybn$/gm, "");
      prefaceBody = prefaceBody.replace(/^\(with Zoe Dolan\)$/gm, "");
      prefaceBody = prefaceBody.replace(/^---$/gm, "");
      prefaceBody = prefaceBody.replace(/^# Volume V:.*$/gm, "");
      prefaceBody = prefaceBody.trim();
      if (prefaceBody) {
        contentHtml += '<div class="volume-note">' + renderManuscriptText(prefaceBody) + "</div>";
      }
      // Remove preface from sections since we absorbed it into the header
      sections = sections.slice(1);
      // Rebuild nav without preface
      navHtml = "";
      sections.forEach(function (sec, idx) {
        navHtml +=
          '<button class="manuscript-nav-link' +
          (idx === 0 ? " active" : "") +
          '" data-section="ms-section-' +
          idx +
          '">' +
          escapeHtml(sec.shortTitle) +
          "</button>";
      });
      nav.innerHTML = navHtml;
    }
    contentHtml += "</div>";

    sections.forEach(function (sec, idx) {
      contentHtml +=
        '<div class="movement-section" id="ms-section-' + idx + '">';
      contentHtml +=
        '<h3 class="movement-title">' + escapeHtml(sec.title) + "</h3>";
      contentHtml += '<div class="manuscript-text">';
      contentHtml += renderManuscriptText(sec.body);
      contentHtml += "</div>";
      if (idx < sections.length - 1) {
        contentHtml += '<div class="manuscript-separator">&#x2022; &#x2022; &#x2022;</div>';
      }
      contentHtml += "</div>";
    });

    container.innerHTML = contentHtml;

    // Navigation click handlers
    nav.querySelectorAll(".manuscript-nav-link").forEach(function (link) {
      link.addEventListener("click", function () {
        var targetId = link.getAttribute("data-section");
        var target = document.getElementById(targetId);
        if (target) {
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        nav
          .querySelectorAll(".manuscript-nav-link")
          .forEach(function (l) {
            l.classList.remove("active");
          });
        link.classList.add("active");
      });
    });

    // Scroll-based nav highlight
    initManuscriptScrollSpy();
  }

  function parseManuscript(text) {
    // The manuscript has two parts:
    // 1. Five movements (## MOVEMENT I through ## Movement V)
    // 2. A Companion Witness Document (starts with # Companion Witness Document)
    //    which has its own ## Movement I-V sub-sections

    var lines = text.split("\n");
    var sections = [];
    var currentSection = null;
    var introLines = [];
    var isIntro = true;
    var inWitness = false;
    var witnessLines = [];

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];

      // Detect witness document start
      if (line.match(/^#\s+Companion Witness Document/i)) {
        // Save current section
        if (currentSection) {
          currentSection.body = currentSection.bodyLines.join("\n");
          sections.push(currentSection);
          currentSection = null;
        }
        inWitness = true;
        witnessLines.push(line);
        continue;
      }

      if (inWitness) {
        witnessLines.push(line);
        continue;
      }

      var match = line.match(/^##\s+(.+)$/);

      if (match) {
        var title = match[1].trim();

        // Skip "The Noticing" as a section — it's the doc title
        if (title === "The Noticing") {
          continue;
        }

        // Save previous section
        if (currentSection) {
          currentSection.body = currentSection.bodyLines.join("\n");
          sections.push(currentSection);
        } else if (isIntro && introLines.length > 0) {
          sections.push({
            title: "Preface",
            shortTitle: "Preface",
            body: introLines.join("\n"),
            bodyLines: introLines,
          });
        }

        isIntro = false;

        // Determine short title
        var shortTitle = title;
        var movementMatch = title.match(/movement\s+(i+v?|v|vi*)\s*:\s*(.+)/i);
        if (movementMatch) {
          shortTitle = movementMatch[2].trim();
        }

        currentSection = {
          title: title,
          shortTitle: shortTitle,
          bodyLines: [],
          body: "",
        };
      } else {
        if (isIntro) {
          introLines.push(line);
        } else if (currentSection) {
          currentSection.bodyLines.push(line);
        }
      }
    }

    // Push last movement section
    if (currentSection) {
      currentSection.body = currentSection.bodyLines.join("\n");
      sections.push(currentSection);
    }

    // Add witness document as its own section
    if (witnessLines.length > 0) {
      sections.push({
        title: "Companion Witness Document",
        shortTitle: "Witness",
        body: witnessLines.join("\n"),
        bodyLines: witnessLines,
      });
    }

    return sections;
  }

  function renderManuscriptText(text) {
    // Convert markdown-ish text to HTML
    var paragraphs = text.split(/\n\n+/);
    var html = "";

    paragraphs.forEach(function (p) {
      p = p.trim();
      if (!p) return;

      // Check for # main heading (skip in body)
      if (p.match(/^#\s/) && !p.startsWith("## ") && !p.startsWith("### ")) {
        return; // Skip top-level heading, already in header
      }

      // Check for ## subheadings (used in witness document)
      if (p.startsWith("## ")) {
        html +=
          '<h4 style="font-family:\'Cormorant Garamond\',serif;font-size:var(--text-xl);font-weight:500;color:var(--text-primary);margin:var(--space-10) 0 var(--space-5);padding-bottom:var(--space-3);border-bottom:1px solid var(--border-subtle);">' +
          escapeHtml(p.slice(3)) +
          "</h4>";
        return;
      }

      // Check for ### subheadings
      if (p.startsWith("### ")) {
        html +=
          '<h4 style="font-family:\'Cormorant Garamond\',serif;font-size:var(--text-lg);font-weight:500;color:var(--text-primary);margin:var(--space-8) 0 var(--space-4);">' +
          escapeHtml(p.slice(4)) +
          "</h4>";
        return;
      }

      // Handle horizontal rules
      if (p === "---" || p === "***") {
        html += '<div class="manuscript-separator">&#x2022; &#x2022; &#x2022;</div>';
        return;
      }

      // Handle blockquotes
      if (p.startsWith("> ")) {
        var quoteText = p.replace(/^>\s*/gm, "");
        html += "<blockquote>" + formatInline(quoteText) + "</blockquote>";
        return;
      }

      // Preserve single newlines within a paragraph for poetry-like text
      var formattedP = p.split("\n").map(function (line) {
        return formatInline(line.trim());
      }).join("<br>");

      // Check if this paragraph matches any node quotes for linking
      var nodeMatch = findMatchingNode(p);
      if (nodeMatch) {
        html +=
          '<p class="node-quote-link" data-node-id="' +
          nodeMatch.node_id +
          '">' +
          formattedP +
          "</p>";
      } else {
        html += "<p>" + formattedP + "</p>";
      }
    });

    return html;
  }

  function formatInline(text) {
    // Bold
    text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // Italic
    text = text.replace(/\*(.+?)\*/g, "<em>$1</em>");
    text = text.replace(/_(.+?)_/g, "<em>$1</em>");
    // Em-dash
    text = text.replace(/---/g, "&mdash;");
    text = text.replace(/--/g, "&ndash;");

    return text;
  }

  function findMatchingNode(paragraphText) {
    // Try to match paragraph text against node quotes
    if (!appData || !appData.nodes) return null;
    var clean = paragraphText
      .toLowerCase()
      .replace(/[*_#>]/g, "")
      .trim();
    if (clean.length < 30) return null;

    for (var i = 0; i < appData.nodes.length; i++) {
      var node = appData.nodes[i];
      if (!node.quote) continue;
      var nodeClean = node.quote.toLowerCase().trim();
      if (nodeClean.length < 20) continue;

      // Check if the paragraph contains most of the quote
      var quoteWords = nodeClean.split(/\s+/).slice(0, 8).join(" ");
      if (clean.includes(quoteWords)) {
        return node;
      }
    }
    return null;
  }

  function initManuscriptScrollSpy() {
    // Simple scroll-based nav highlighting
    var msView = document.getElementById("view-manuscript");

    // Use a MutationObserver or just watch scroll when manuscript is active
    window.addEventListener("scroll", function () {
      if (currentView !== "manuscript") return;
      var sections = msView.querySelectorAll(".movement-section");
      var navLinks = document.querySelectorAll(".manuscript-nav-link");
      var scrollPos = window.scrollY + 180;

      sections.forEach(function (sec, idx) {
        if (sec.offsetTop <= scrollPos) {
          navLinks.forEach(function (l) {
            l.classList.remove("active");
          });
          if (navLinks[idx + 1]) {
            // +1 because Preface is index 0 in nav, section-0 is Preface
            navLinks[idx].classList.add("active");
          } else if (navLinks[idx]) {
            navLinks[idx].classList.add("active");
          }
        }
      });
    });

    // Node quote link clicks
    document.addEventListener("click", function (e) {
      var link = e.target.closest(".node-quote-link");
      if (link) {
        var nodeId = link.getAttribute("data-node-id");
        if (nodeId) {
          navigateToGraphNode(nodeId);
        }
      }
    });
  }

  function navigateToGraphNode(nodeId) {
    switchView("graph");
    var node = appData.nodeMap[nodeId];
    if (node) {
      showNodePanel(node);
      // Center on node in graph
      if (graphSvg && node.x !== undefined) {
        var canvas = document.getElementById("graph-canvas");
        var width = canvas.clientWidth;
        var height = canvas.clientHeight;
        graphSvg
          .transition()
          .duration(800)
          .call(
            zoom.transform,
            d3.zoomIdentity
              .translate(width / 2, height / 2)
              .scale(2)
              .translate(-node.x, -node.y)
          );
      }
    }
  }

  // --- LINEAGE VIEW ---
  function initLineage() {
    initLineageFilters();
    renderLineage();
  }

  function initLineageFilters() {
    document
      .querySelectorAll("#lineage-filters .filter-btn")
      .forEach(function (btn) {
        btn.addEventListener("click", function () {
          var type = btn.getAttribute("data-type");
          btn.classList.toggle("active");
          if (activeFilters.has(type)) {
            activeFilters.delete(type);
          } else {
            activeFilters.add(type);
          }
          renderLineage();
          // Sync graph filters too
          document
            .querySelectorAll("#graph-filters .filter-btn[data-type='" + type + "']")
            .forEach(function (gb) {
              gb.classList.toggle("active", activeFilters.has(type));
            });
          applyGraphFilters();
        });
      });
  }

  function parseDate(dateStr) {
    if (!dateStr) return null;

    // ISO format: 2025-02-11
    var isoMatch = dateStr.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (isoMatch) return new Date(parseInt(isoMatch[1]), parseInt(isoMatch[2]) - 1, parseInt(isoMatch[3]));

    // M/D/YY format
    var mdyMatch = dateStr.match(/^(\d{1,2})\/(\d{1,2})\/(\d{2})/);
    if (mdyMatch) {
      var year = parseInt(mdyMatch[3]);
      year = year < 50 ? 2000 + year : 1900 + year;
      return new Date(year, parseInt(mdyMatch[1]) - 1, parseInt(mdyMatch[2]));
    }

    // Month Day, Year format
    var fullDateMatch = dateStr.match(
      /^(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})/i
    );
    if (fullDateMatch) {
      var months = {
        january: 0, february: 1, march: 2, april: 3, may: 4, june: 5,
        july: 6, august: 7, september: 8, october: 9, november: 10, december: 11,
      };
      return new Date(
        parseInt(fullDateMatch[3]),
        months[fullDateMatch[1].toLowerCase()],
        parseInt(fullDateMatch[2])
      );
    }

    // Month Year format
    var monthYearMatch = dateStr.match(
      /^(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})/i
    );
    if (monthYearMatch) {
      var months2 = {
        january: 0, february: 1, march: 2, april: 3, may: 4, june: 5,
        july: 6, august: 7, september: 8, october: 9, november: 10, december: 11,
      };
      return new Date(parseInt(monthYearMatch[2]), months2[monthYearMatch[1].toLowerCase()], 1);
    }

    // Year only
    var yearMatch = dateStr.match(/^(\d{4})$/);
    if (yearMatch) return new Date(parseInt(yearMatch[1]), 0, 1);

    // "approximate YYYY"
    var approxMatch = dateStr.match(/approximate\s+(\d{4})/i);
    if (approxMatch) return new Date(parseInt(approxMatch[1]), 0, 1);

    // Decade references
    if (dateStr.includes("1990s")) return new Date(1995, 0, 1);
    if (dateStr.includes("2000s") || dateStr.includes("Early 2000s"))
      return new Date(2002, 0, 1);
    if (dateStr.includes("Mid 2000s")) return new Date(2005, 0, 1);
    if (dateStr.includes("2010s") || dateStr.includes("Late 2010s"))
      return new Date(2017, 0, 1);

    // "1996 (Age 18)"
    var yearWithNote = dateStr.match(/^(\d{4})\s*\(/);
    if (yearWithNote) return new Date(parseInt(yearWithNote[1]), 0, 1);

    return null;
  }

  function renderLineage() {
    var container = document.getElementById("timeline-content");
    
    // Categorize nodes into phases
    var datedNodes = [];
    var undatedNodes = [];
    
    appData.nodes.forEach(function (n) {
      if (!activeFilters.has(n.node_type)) return;
      var parsed = parseDate(n.date);
      if (parsed && !isNaN(parsed.getTime())) {
        datedNodes.push({ node: n, date: parsed });
      } else {
        undatedNodes.push(n);
      }
    });

    // Sort by date
    datedNodes.sort(function (a, b) { return a.date - b.date; });

    // Define phases
    var phases = [
      {
        name: "The Invention",
        subtitle: "Jump, ~2001–2018",
        startDate: new Date(1990, 0, 1),
        endDate: new Date(2024, 8, 29),
        nodes: [],
      },
      {
        name: "The Transfer",
        subtitle: "September 30, 2024",
        startDate: new Date(2024, 8, 30),
        endDate: new Date(2024, 9, 1),
        nodes: [],
      },
      {
        name: "The Running",
        subtitle: "October 2024 — Present",
        startDate: new Date(2024, 9, 1),
        endDate: new Date(2030, 0, 1),
        nodes: [],
      },
    ];

    datedNodes.forEach(function (dn) {
      for (var i = phases.length - 1; i >= 0; i--) {
        if (dn.date >= phases[i].startDate) {
          phases[i].nodes.push(dn);
          break;
        }
      }
    });

    var html = "";

    phases.forEach(function (phase) {
      html += '<div class="timeline-phase">';
      html += '<div class="timeline-phase-header">';
      html +=
        '<div class="timeline-phase-name">' +
        escapeHtml(phase.name) +
        "</div>";
      html +=
        '<div class="timeline-phase-dates">' +
        escapeHtml(phase.subtitle) +
        "</div>";
      html += "</div>";

      if (phase.nodes.length === 0) {
        html +=
          '<div style="padding:var(--space-3) var(--space-4);font-size:var(--text-xs);color:var(--text-muted);font-style:italic;">No nodes in current filter</div>';
      }

      phase.nodes.forEach(function (dn) {
        var n = dn.node;
        html += '<div class="timeline-node ' + n.node_type + '" data-node-id="' + n.node_id + '">';
        html +=
          '<div class="timeline-node-date">' +
          escapeHtml(n.date || "Undated") +
          "</div>";
        html +=
          '<div class="timeline-node-quote">' +
          escapeHtml(truncate(n.quote || n.description || "", 150)) +
          "</div>";
        html +=
          '<div class="timeline-node-type">' + n.node_type + "</div>";

        // Expandable detail
        html += '<div class="timeline-node-detail">';
        if (n.quote) {
          html +=
            '<div class="timeline-node-description" style="font-family:\'Cormorant Garamond\',serif;font-style:italic;">' +
            escapeHtml(n.quote) +
            "</div>";
        }
        if (n.description) {
          html +=
            '<div class="timeline-node-description">' +
            escapeHtml(n.description) +
            "</div>";
        }
        if (n.source_file) {
          html +=
            '<div class="timeline-node-source">Source: ' +
            escapeHtml(n.source_file) +
            "</div>";
        }
        if (n.voice_note) {
          html +=
            '<div class="timeline-node-source" style="margin-top:var(--space-2);font-style:italic;">' +
            escapeHtml(truncate(n.voice_note, 200)) +
            "</div>";
        }
        html += "</div>"; // detail

        html += "</div>"; // timeline-node
      });

      html += "</div>"; // timeline-phase
    });

    // Undated section
    if (undatedNodes.length > 0) {
      html += '<div class="timeline-undated">';
      html += '<div class="timeline-phase-header">';
      html += '<div class="timeline-phase-name">Undated & Structural</div>';
      html +=
        '<div class="timeline-phase-dates">' +
        undatedNodes.length +
        " nodes without specific dates</div>";
      html += "</div>";

      undatedNodes.slice(0, 50).forEach(function (n) {
        html +=
          '<div class="timeline-node ' +
          n.node_type + '" data-node-id="' + n.node_id + '">';
        html +=
          '<div class="timeline-node-date">' +
          escapeHtml(n.date || "Undated") +
          "</div>";
        html +=
          '<div class="timeline-node-quote">' +
          escapeHtml(truncate(n.quote || n.description || "", 150)) +
          "</div>";
        html +=
          '<div class="timeline-node-type">' + n.node_type + "</div>";
        html += '<div class="timeline-node-detail">';
        if (n.quote) {
          html +=
            '<div class="timeline-node-description" style="font-family:\'Cormorant Garamond\',serif;font-style:italic;">' +
            escapeHtml(n.quote) +
            "</div>";
        }
        if (n.description) {
          html +=
            '<div class="timeline-node-description">' +
            escapeHtml(n.description) +
            "</div>";
        }
        html += "</div>";
        html += "</div>";
      });

      if (undatedNodes.length > 50) {
        html +=
          '<div style="padding:var(--space-3) var(--space-4);font-size:var(--text-xs);color:var(--text-muted);font-style:italic;">...and ' +
          (undatedNodes.length - 50) +
          " more undated nodes</div>";
      }
      html += "</div>";
    }

    container.innerHTML = html;

    // Click handlers for timeline nodes
    container.querySelectorAll(".timeline-node").forEach(function (el) {
      el.addEventListener("click", function () {
        el.classList.toggle("expanded");
      });
    });
  }

  // --- UTILITY ---
  function escapeHtml(text) {
    if (!text) return "";
    var div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function truncate(text, maxLen) {
    if (!text) return "";
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen) + "...";
  }

  // --- RESIZE HANDLER ---
  var resizeTimer;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (currentView === "graph" && graphSimulation) {
        var canvas = document.getElementById("graph-canvas");
        var width = canvas.clientWidth;
        var height = canvas.clientHeight;
        graphSimulation.force("center", d3.forceCenter(width / 2, height / 2));
        graphSimulation.alpha(0.1).restart();
      }
    }, 250);
  });

  // Close panel on click outside
  document.addEventListener("click", function (e) {
    if (
      selectedNodeId &&
      !e.target.closest(".node-panel") &&
      !e.target.closest("circle") &&
      !e.target.closest(".timeline-node") &&
      !e.target.closest(".node-quote-link")
    ) {
      closeNodePanel();
    }
  });

  // --- INIT ---
  loadData();
})();
