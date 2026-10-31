// ============================================================
// CLASSIFICATION PERFORMANCE
// ============================================================

let performanceData =
  null;


let activePerformanceMetric =
  "accuracy";


const performanceMetricLabels = {

  accuracy:
    "Accuracy",

  sensitivity:
    "Sensitivity",

  specificity:
    "Specificity",

  f1_score:
    "F1-Score",

  precision:
    "Precision"

};


const performanceMetricNotes = {

  accuracy:
    "Bars represent classification accuracy (%).",

  sensitivity:
    "Bars represent sensitivity (%), measuring the proportion of Alzheimer's disease cases correctly identified.",

  specificity:
    "Bars represent specificity (%), measuring the proportion of cognitively normal cases correctly identified.",

  f1_score:
    "Bars represent F1-Score (%), balancing precision and sensitivity.",

  precision:
    "Bars represent precision (%), measuring the proportion of Alzheimer's disease predictions that were correct."

};


const performanceLlmMethodOrder = [
  "Zero-shot",
  "Random",
  "Cosine",
  "Euclidean",
  "Delta-KNN",
  "Proposed"
];



// ============================================================
// LOAD PERFORMANCE DATA
// ============================================================

fetch(
  "data/accuracy_results.json"
)

  .then(response => {

    if (!response.ok) {

      throw new Error(
        `HTTP error: ${response.status}`
      );

    }

    return response.json();

  })


  .then(data => {

    performanceData =
      data;


    renderPerformanceChart();

  })


  .catch(error => {

    console.error(
      "Could not load performance data:",
      error
    );


    const container =
      document.getElementById(
        "accuracy-chart"
      );


    if (container) {

      container.innerHTML =
        "<p>Could not load classification performance data.</p>";

    }

  });



// ============================================================
// PERFORMANCE METRIC TABS
// ============================================================

const performanceMetricTabs =
  document.querySelectorAll(
    ".performance-metric-tab"
  );


performanceMetricTabs.forEach(
  tab => {

    tab.addEventListener(
      "click",
      () => {


        activePerformanceMetric =
          tab.dataset.performanceMetric;


        performanceMetricTabs.forEach(
          item => {


            const active =
              item === tab;


            item.classList.toggle(
              "active",
              active
            );


            item.setAttribute(
              "aria-selected",
              String(active)
            );

          }
        );


        renderPerformanceChart();

      }
    );

  }
);



// ============================================================
// GET PERFORMANCE GROUPS
// ============================================================

function getPerformanceGroups() {


  if (!performanceData) {

    return [];

  }


  const metricData =
    performanceData
      .metrics?.[
        activePerformanceMetric
      ];


  if (!metricData) {

    return [];

  }


  const groups = [];



  // ==========================================================
  // MACHINE LEARNING
  // ==========================================================

  const mlData =
    metricData.ml || {};


  groups.push({

    title:
      "Machine Learning",

    values:
      Object.entries(
        mlData
      ).map(
        (
          [
            method,
            value
          ]
        ) => ({

          method:
            method,

          value:
            value === null
              ? null
              : Number(value),

          proposed:
            false

        })
      )

  });



  // ==========================================================
  // GPT-5.4
  // ==========================================================

  const gptData =
    metricData
      .llm?.["GPT-5.4"]
      || {};


  groups.push({

    title:
      "GPT-5.4",

    values:
      performanceLlmMethodOrder
        .filter(
          method =>
            Object.prototype
              .hasOwnProperty
              .call(
                gptData,
                method
              )
        )
        .map(
          method => ({

            method:
              method,

            value:
              gptData[method] === null
                ? null
                : Number(
                    gptData[
                      method
                    ]
                  ),

            proposed:
              method
              === "Proposed"

          })
        )

  });



  // ==========================================================
  // GROK-4.2
  // ==========================================================

  const grokData =
    metricData
      .llm?.["Grok-4.2"]
      || {};


  groups.push({

    title:
      "Grok-4.2",

    values:
      performanceLlmMethodOrder
        .filter(
          method =>
            Object.prototype
              .hasOwnProperty
              .call(
                grokData,
                method
              )
        )
        .map(
          method => ({

            method:
              method,

            value:
              grokData[method] === null
                ? null
                : Number(
                    grokData[
                      method
                    ]
                  ),

            proposed:
              method
              === "Proposed"

          })
        )

  });


  return groups;

}



// ============================================================
// RENDER PERFORMANCE CHART
// ============================================================

function renderPerformanceChart() {


  const container =
    document.getElementById(
      "accuracy-chart"
    );


  if (
    !container
    || !performanceData
  ) {

    return;

  }


  const groups =
    getPerformanceGroups();


  if (
    groups.length === 0
  ) {

    container.innerHTML =
      "<p>No performance data available.</p>";

    return;

  }


  const metricLabel =
    performanceMetricLabels[
      activePerformanceMetric
    ];


  container.innerHTML =
    "";


  const wrapper =
    document.createElement(
      "div"
    );


  wrapper.className =
    "accuracy-groups";


  groups.forEach(
    (
      group,
      groupIndex
    ) => {


      const groupElement =
        document.createElement(
          "div"
        );


      groupElement.className =
        "accuracy-group";


      if (
        groupIndex > 0
      ) {

        groupElement.classList.add(
          "accuracy-group-separated"
        );

      }



      // ========================================================
      // GROUP HEADING
      // ========================================================

      const heading =
        document.createElement(
          "div"
        );


      heading.className =
        "accuracy-group-heading";


      heading.innerHTML = `

        <h4>
          ${group.title}
        </h4>

        <span>
          ${metricLabel} (%)
        </span>

      `;


      groupElement.appendChild(
        heading
      );



      // ========================================================
      // BARS
      // ========================================================

      const bars =
        document.createElement(
          "div"
        );


      bars.className =
        "accuracy-bars";


      group.values.forEach(
        item => {


          const hasValue =
            Number.isFinite(
              item.value
            );


          const displayValue =
            hasValue
              ? Math.max(
                  0,
                  Math.min(
                    100,
                    item.value
                  )
                )
              : 0;


          const displayText =
            hasValue
              ? `${displayValue.toFixed(2)}%`
              : "—";



          const row =
            document.createElement(
              "div"
            );


          row.className =
            "accuracy-row";


          if (
            item.proposed
          ) {

            row.classList.add(
              "accuracy-row-proposed"
            );

          }



          row.innerHTML = `

            <div
              class="
                accuracy-method
                ${
                  item.proposed
                    ? "accuracy-method-proposed"
                    : ""
                }
              "
            >

              ${
                item.proposed
                  ? '<span class="proposed-star">★</span>'
                  : ""
              }

              <span>
                ${item.method}
              </span>

            </div>


            <div class="accuracy-bar-area">

              <div class="accuracy-bar-track">

                <div
                  class="
                    accuracy-bar-fill

                    ${
                      item.proposed
                        ? "accuracy-bar-proposed"
                        : ""
                    }

                    ${
                      !hasValue
                        ? "accuracy-bar-missing"
                        : ""
                    }
                  "

                  style="
                    width:
                      ${displayValue}%;
                  "
                ></div>

              </div>


              <div class="accuracy-value">
                ${displayText}
              </div>

            </div>

          `;


          bars.appendChild(
            row
          );

        }
      );


      groupElement.appendChild(
        bars
      );


      wrapper.appendChild(
        groupElement
      );

    }
  );


  container.appendChild(
    wrapper
  );



  // ==========================================================
  // UPDATE NOTE
  // ==========================================================

  const note =
    document.getElementById(
      "performance-metric-note"
    );


  if (note) {

    note.textContent =
      `${performanceMetricNotes[
        activePerformanceMetric
      ]} Machine-learning models are shown first, followed by GPT-5.4 and Grok-4.2. All panels use the same 0–100% scale to support direct comparison.`;

  }

}


// ============================================================
// REASONING CONSISTENCY
// ============================================================

fetch(
  "data/consistency.json"
)

  .then(response => {

    if (!response.ok) {

      throw new Error(
        `HTTP error: ${response.status}`
      );

    }

    return response.json();

  })

  .then(data => {


    const container =
      document.getElementById(
        "consistency-grid"
      );


    if (!container) {
      return;
    }


    const models =
      Object.keys(
        data
      );


    const preferredOrder = [
      "Proposed",
      "Zero-shot",
      "Random",
      "Cosine",
      "Euclidean",
      "Delta-KNN"
    ];


    // --------------------------------------------------------
    // Collect methods available across all models
    // --------------------------------------------------------

    const availableMethods =
      new Set();


    models.forEach(model => {

      Object.keys(
        data[model]
      ).forEach(method => {

        availableMethods.add(
          method
        );

      });

    });


    const methods =
      preferredOrder.filter(
        method =>
          availableMethods.has(
            method
          )
      );


    // --------------------------------------------------------
    // Maximum count for colour intensity
    // --------------------------------------------------------

    const allValues = [];


    models.forEach(model => {

      methods.forEach(method => {

        if (!data[model][method]) {
          return;
        }

        data[model][method]
          .flat()
          .forEach(value => {

            allValues.push(
              Number(value)
            );

          });

      });

    });


    const maxValue =
      Math.max(
        ...allValues,
        1
      );


    // --------------------------------------------------------
    // Build grid
    // --------------------------------------------------------

    const grid =
      document.createElement(
        "div"
      );


    grid.className =
      "consistency-grid";


    grid.style.gridTemplateColumns =
      `120px repeat(
        ${methods.length},
        minmax(130px, 1fr)
      )`;


    let html =
      `<div></div>`;


    // --------------------------------------------------------
    // Method headings
    // --------------------------------------------------------

    methods.forEach(method => {

      html += `
        <div class="method-heading">
          ${method}
        </div>
      `;

    });


    // --------------------------------------------------------
    // Models
    // --------------------------------------------------------

    models.forEach(
      (
        model,
        modelIndex
      ) => {


        html += `
          <div class="model-axis">

            <div class="model-name">
              ${model}
            </div>

            <div class="prediction-axis-title">
              Prediction
            </div>

            <div class="prediction-axis-labels">
              <span>C</span>
              <span>I</span>
            </div>

          </div>
        `;


        methods.forEach(method => {


          const matrix =
            data[model][method];


          // If one method does not exist for one model,
          // show an empty placeholder.
          if (!matrix) {

            html += `
              <div class="matrix-wrapper">

                <div class="compact-matrix matrix-empty">
                  —
                </div>

              </div>
            `;

            return;
          }


          const cc =
            Number(
              matrix[0][0]
            );

          const ci =
            Number(
              matrix[0][1]
            );

          const ic =
            Number(
              matrix[1][0]
            );

          const ii =
            Number(
              matrix[1][1]
            );


          const opacity =
            value =>
              0.12
              + (
                value
                / maxValue
              )
              * 0.75;


          html += `
            <div class="matrix-wrapper">

              <div class="compact-matrix">

                <div
                  class="matrix-cell"
                  style="
                    background:
                    rgba(
                      190,
                      145,
                      40,
                      ${opacity(cc)}
                    )
                  "
                  title="
                    Prediction consistent,
                    reasoning consistent:
                    ${cc}
                  "
                >
                  ${cc}
                </div>


                <div
                  class="matrix-cell"
                  style="
                    background:
                    rgba(
                      190,
                      145,
                      40,
                      ${opacity(ci)}
                    )
                  "
                  title="
                    Prediction consistent,
                    reasoning inconsistent:
                    ${ci}
                  "
                >
                  ${ci}
                </div>


                <div
                  class="matrix-cell"
                  style="
                    background:
                    rgba(
                      190,
                      145,
                      40,
                      ${opacity(ic)}
                    )
                  "
                  title="
                    Prediction inconsistent,
                    reasoning consistent:
                    ${ic}
                  "
                >
                  ${ic}
                </div>


                <div
                  class="matrix-cell"
                  style="
                    background:
                    rgba(
                      190,
                      145,
                      40,
                      ${opacity(ii)}
                    )
                  "
                  title="
                    Prediction inconsistent,
                    reasoning inconsistent:
                    ${ii}
                  "
                >
                  ${ii}
                </div>

              </div>


              ${
                modelIndex
                === models.length - 1

                  ? `
                    <div class="reasoning-axis-labels">
                      <span>C</span>
                      <span>I</span>
                    </div>

                    <div class="reasoning-axis-title">
                      Reasoning
                    </div>
                  `

                  : ""
              }

            </div>
          `;

        });

      }
    );


    grid.innerHTML =
      html;


    container.innerHTML =
      "";


    container.appendChild(
      grid
    );


    // --------------------------------------------------------
    // Legend
    // --------------------------------------------------------

    const legend =
      document.createElement(
        "div"
      );


    legend.className =
      "consistency-legend";


    legend.innerHTML = `
      <strong>C</strong>
      = Consistent

      &nbsp;&nbsp;&nbsp;

      <strong>I</strong>
      = Inconsistent
    `;


    container.appendChild(
      legend
    );

  })


  .catch(error => {

    console.error(
      "Could not load consistency data:",
      error
    );


    const container =
      document.getElementById(
        "consistency-grid"
      );


    if (container) {

      container.innerHTML =
        "<p>Could not load consistency data.</p>";

    }

  });

// ============================================================
// REGIONAL INTERPRETABILITY HEATMAP
//
// Layout:
//
// Brain region
// |
// Logistic Regression
// XGBoost
// |
// Zero-shot
// Random
// Cosine
// Euclidean
// Delta-KNN
// Proposed
//
// The selected LLM controls the region set used by both
// the ML and LLM parts of the heatmap.
// ============================================================


const regionMethodOrder = [
  "Zero-shot",
  "Random",
  "Cosine",
  "Euclidean",
  "Delta-KNN",
  "Proposed"
];


const regionMlOrder = [
  "Logistic Regression",
  "XGBoost"
];


let regionInterpretabilityData =
  null;


let activeRegionModel =
  "GPT-5.4";



// ============================================================
// LOAD REGIONAL INTERPRETABILITY DATA
// ============================================================

fetch(
  "data/region_interpretability.json"
)

  .then(response => {

    if (!response.ok) {

      throw new Error(
        `HTTP error: ${response.status}`
      );

    }

    return response.json();

  })


  .then(data => {

    regionInterpretabilityData =
      data;


    renderRegionHeatmap(
      activeRegionModel
    );

  })


  .catch(error => {

    console.error(
      "Could not load regional interpretability data:",
      error
    );


    const container =
      document.getElementById(
        "region-heatmap"
      );


    if (container) {

      container.innerHTML =
        "<p>Could not load regional interpretability data.</p>";

    }

  });



// ============================================================
// REGION MODEL TABS
// ============================================================

const regionTabs =
  document.querySelectorAll(
    ".region-tab"
  );


regionTabs.forEach(tab => {

  tab.addEventListener(
    "click",
    () => {


      activeRegionModel =
        tab.dataset.regionModel;


      regionTabs.forEach(
        item => {


          const active =
            item === tab;


          item.classList.toggle(
            "active",
            active
          );


          item.setAttribute(
            "aria-selected",
            String(active)
          );

        }
      );


      renderRegionHeatmap(
        activeRegionModel
      );

    }
  );

});



// ============================================================
// LLM CELL BACKGROUND
//
// Blue = mean subject-level mention frequency
// ============================================================

function getRegionCellColor(
  value
) {


  const numericValue =
    Math.max(
      0,
      Math.min(
        100,
        Number(value)
      )
    );


  const opacity =
    0.08
    + (
      numericValue
      / 100
    )
      * 0.82;


  return (
    `rgba(
      49,
      87,
      213,
      ${opacity}
    )`
  );

}



// ============================================================
// ML CELL BACKGROUND
//
// Green = relative regional importance
// ============================================================

function getMlRegionCellColor(
  value
) {

  const numericValue =
    Math.max(
      0,
      Math.min(
        100,
        Number(value)
      )
    );


  const opacity =
    0.08
    + (
      numericValue
      / 100
    )
      * 0.82;


  return (
    `rgba(
      124,
      92,
      191,
      ${opacity}
    )`
  );

}



// ============================================================
// CELL TEXT COLOUR
// ============================================================

function getRegionTextColor(
  value
) {


  return (
    Number(value) >= 65
      ? "#ffffff"
      : "#172033"
  );

}



// ============================================================
// RENDER REGIONAL INTERPRETABILITY HEATMAP
// ============================================================

function renderRegionHeatmap(
  model
) {


  const container =
    document.getElementById(
      "region-heatmap"
    );


  if (
    !container
    || !regionInterpretabilityData
  ) {

    return;

  }


  // ----------------------------------------------------------
  // Expected JSON:
  //
  // {
  //   "models": {
  //
  //     "GPT-5.4": {
  //
  //       "regions": [
  //         "Hippocampus",
  //         ...
  //       ],
  //
  //       "ml": {
  //
  //         "Logistic Regression": {
  //           "Hippocampus": 100
  //         },
  //
  //         "XGBoost": {
  //           "Hippocampus": 90
  //         }
  //
  //       },
  //
  //       "llm": {
  //
  //         "Zero-shot": {
  //           "Hippocampus": 80
  //         },
  //
  //         ...
  //
  //       }
  //
  //     }
  //
  //   }
  // }
  // ----------------------------------------------------------


  const modelData =
    regionInterpretabilityData
      .models?.[model];


  if (!modelData) {


    container.innerHTML =
      `<p>
        No regional interpretability data available
        for ${model}.
      </p>`;


    return;

  }



  // ==========================================================
  // REGIONS
  //
  // These are already ordered in the JSON according to mean
  // LLM mention frequency.
  //
  // All regions are retained.
  // ==========================================================

  const regions =
    Array.isArray(
      modelData.regions
    )
      ? modelData.regions
      : [];



  // ==========================================================
  // AVAILABLE ML MODELS
  // ==========================================================

  const mlModels =
    regionMlOrder.filter(
      mlModel =>
        Object.prototype
          .hasOwnProperty
          .call(
            modelData.ml || {},
            mlModel
          )
    );



  // ==========================================================
  // AVAILABLE LLM METHODS
  // ==========================================================

  const methods =
    regionMethodOrder.filter(
      method =>
        Object.prototype
          .hasOwnProperty
          .call(
            modelData.llm || {},
            method
          )
    );



  if (
    regions.length === 0
  ) {


    container.innerHTML =
      `<p>
        No brain-region data available
        for ${model}.
      </p>`;


    return;

  }



  // ==========================================================
  // CREATE GRID
  // ==========================================================

  const grid =
    document.createElement(
      "div"
    );


  grid.className =
    "region-heatmap-grid";


  // ----------------------------------------------------------
  // Columns:
  //
  // brain region
  // + ML columns
  // + separator
  // + LLM columns
  // ----------------------------------------------------------

  grid.style.gridTemplateColumns =
    `
      minmax(190px, 1.6fr)

      repeat(
        ${mlModels.length},
        minmax(105px, 1fr)
      )

      3px

      repeat(
        ${methods.length},
        minmax(92px, 1fr)
      )
    `;



  // ==========================================================
  // HEADER ROW
  // ==========================================================

  let html = `

    <div class="region-corner">
      Brain region
    </div>

  `;



  // ==========================================================
  // ML HEADINGS
  // ==========================================================

  mlModels.forEach(
    mlModel => {


      html += `

        <div
          class="
            region-method-heading
            region-ml-heading
          "
        >
          ${mlModel}
        </div>

      `;

    }
  );



  // ==========================================================
  // GROUP SEPARATOR
  // ==========================================================

  html += `

    <div
      class="region-group-separator"
      aria-hidden="true"
    ></div>

  `;



  // ==========================================================
  // LLM METHOD HEADINGS
  // ==========================================================

  methods.forEach(
    method => {


      html += `

        <div
          class="
            region-method-heading
            region-llm-heading
          "
        >
          ${method}
        </div>

      `;

    }
  );



  // ==========================================================
  // REGION ROWS
  // ==========================================================

  regions.forEach(
    region => {


      // -------------------------------------------------------
      // REGION NAME
      // -------------------------------------------------------

      html += `

        <div class="region-name">
          ${region}
        </div>

      `;



      // ======================================================
      // ML VALUES FIRST
      // ======================================================

      mlModels.forEach(
        mlModel => {


          const rawValue =
            modelData
              .ml?.[mlModel]?.[region];


          const value =
            rawValue === undefined
              ? 0
              : Number(
                  rawValue
                );


          const displayValue =
            value.toFixed(
              1
            );


          html += `

            <div
              class="
                region-cell
                region-ml-cell
              "

              style="
                background:
                  ${getMlRegionCellColor(
                    value
                  )};

                color:
                  ${getRegionTextColor(
                    value
                  )};
              "

              title="
                ${region}
                — ${mlModel}:
                ${displayValue}%
                relative regional importance
              "
            >

              ${displayValue}

            </div>

          `;

        }
      );



      // ======================================================
      // VERTICAL GROUP SEPARATOR
      // ======================================================

      html += `

        <div
          class="region-group-separator"
          aria-hidden="true"
        ></div>

      `;



      // ======================================================
      // LLM VALUES
      // ======================================================

      methods.forEach(
        method => {


          const rawValue =
            modelData
              .llm?.[method]?.[region];


          const value =
            rawValue === undefined
              ? 0
              : Number(
                  rawValue
                );


          const displayValue =
            value.toFixed(
              1
            );


          html += `

            <div
              class="
                region-cell
                region-llm-cell
              "

              style="
                background:
                  ${getRegionCellColor(
                    value
                  )};

                color:
                  ${getRegionTextColor(
                    value
                  )};
              "

              title="
                ${region}
                — ${method}:
                ${displayValue}%
                mean subject-level mention frequency
              "
            >

              ${displayValue}

            </div>

          `;

        }
      );

    }
  );



  // ==========================================================
  // INSERT GRID
  // ==========================================================

  grid.innerHTML =
    html;


  container.innerHTML =
    "";


  container.appendChild(
    grid
  );

}



// ============================================================
// DIRECTIONALITY SANKEY
// ============================================================

let sankeyData = null;


const sankeyMethodOrder = [
  "Zero-shot",
  "Random",
  "Cosine",
  "Euclidean",
  "Delta-KNN",
  "Proposed"
];


const sankeyModelSelect =
  document.getElementById(
    "sankey-model"
  );


const sankeyMethodSelect =
  document.getElementById(
    "sankey-method"
  );


// ============================================================
// LOAD DATA
// ============================================================

fetch(
  "data/direction_sankey.json"
)

  .then(response => {

    if (!response.ok) {

      throw new Error(
        `HTTP error: ${response.status}`
      );

    }

    return response.json();

  })

  .then(data => {

    sankeyData = data;

    updateSankeyMethods();

  })

  .catch(error => {

    console.error(
      "Could not load Sankey data:",
      error
    );


    const container =
      document.getElementById(
        "direction-sankey"
      );


    if (container) {

      container.innerHTML =
        "<p>Could not load directionality data.</p>";

    }

  });


// ============================================================
// MODEL CHANGE
// ============================================================

if (sankeyModelSelect) {

  sankeyModelSelect.addEventListener(
    "change",
    () => {

      updateSankeyMethods();

    }
  );

}


// ============================================================
// METHOD CHANGE
// ============================================================

if (sankeyMethodSelect) {

  sankeyMethodSelect.addEventListener(
    "change",
    () => {

      renderDirectionSankey();

    }
  );

}


// ============================================================
// UPDATE AVAILABLE METHODS
// ============================================================

function updateSankeyMethods() {

  if (
    !sankeyData
    || !sankeyModelSelect
    || !sankeyMethodSelect
  ) {

    return;

  }


  const model =
    sankeyModelSelect.value;


  const modelData =
    sankeyData.models?.[model];


  if (!modelData) {

    sankeyMethodSelect.innerHTML = "";

    return;

  }


  const availableMethods =
    sankeyMethodOrder.filter(
      method =>
        Object.prototype
          .hasOwnProperty
          .call(
            modelData,
            method
          )
    );


  sankeyMethodSelect.innerHTML =
    availableMethods
      .map(
        method =>
          `<option value="${method}">
             ${method}
           </option>`
      )
      .join("");


  // Prefer Proposed when available
  if (
    availableMethods.includes(
      "Proposed"
    )
  ) {

    sankeyMethodSelect.value =
      "Proposed";

  }


  renderDirectionSankey();

}


// ============================================================
// CREATE UNIQUE NODE ID
//
// Prefixing is important because a region and direction could
// theoretically have the same text.
// ============================================================

function sankeyNodeId(
  type,
  label
) {

  return `${type}::${label}`;

}


// ============================================================
// RENDER SANKEY
// ============================================================

function renderDirectionSankey() {

  const container =
    document.getElementById(
      "direction-sankey"
    );


  if (
    !container
    || !sankeyData
    || !sankeyModelSelect
    || !sankeyMethodSelect
  ) {

    return;

  }


  const model =
    sankeyModelSelect.value;


  const method =
    sankeyMethodSelect.value;


  const flows =
    sankeyData
      .models?.[model]?.[method];


  if (
    !flows
    || flows.length === 0
  ) {

    container.innerHTML =
      "<p>No Sankey data available for this selection.</p>";

    return;

  }


  container.innerHTML = "";


  // ==========================================================
  // AGGREGATE REGION -> DIRECTION LINKS
  // ==========================================================

  const regionDirectionMap =
    new Map();


  // ==========================================================
  // AGGREGATE DIRECTION -> PREDICTION LINKS
  // ==========================================================

  const directionPredictionMap =
    new Map();


  const nodesMap =
    new Map();


  flows.forEach(flow => {

    const region =
      flow.region;

    const direction =
      flow.direction;

    const prediction =
      flow.prediction;

    const count =
      Number(
        flow.count
      );


    if (
      !Number.isFinite(count)
      || count <= 0
    ) {

      return;

    }


    const regionId =
      sankeyNodeId(
        "region",
        region
      );


    const directionId =
      sankeyNodeId(
        "direction",
        direction
      );


    const predictionId =
      sankeyNodeId(
        "prediction",
        prediction
      );


    // --------------------------------------------------------
    // Nodes
    // --------------------------------------------------------

    nodesMap.set(
      regionId,
      {
        id: regionId,
        name: region,
        type: "region"
      }
    );


    nodesMap.set(
      directionId,
      {
        id: directionId,
        name: direction,
        type: "direction"
      }
    );


    nodesMap.set(
      predictionId,
      {
        id: predictionId,
        name: prediction,
        type: "prediction"
      }
    );


    // --------------------------------------------------------
    // Region -> Direction
    // --------------------------------------------------------

    const rdKey =
      `${regionId}|${directionId}`;


    regionDirectionMap.set(
      rdKey,
      (
        regionDirectionMap.get(
          rdKey
        ) || 0
      ) + count
    );


    // --------------------------------------------------------
    // Direction -> Prediction
    // --------------------------------------------------------

    const dpKey =
      `${directionId}|${predictionId}`;


    directionPredictionMap.set(
      dpKey,
      (
        directionPredictionMap.get(
          dpKey
        ) || 0
      ) + count
    );

  });


  // ==========================================================
  // BUILD LINKS
  // ==========================================================

  const links = [];


  regionDirectionMap.forEach(
    (
      value,
      key
    ) => {

      const [
        source,
        target
      ] =
        key.split("|");


      links.push(
        {
          source,
          target,
          value
        }
      );

    }
  );


  directionPredictionMap.forEach(
    (
      value,
      key
    ) => {

      const [
        source,
        target
      ] =
        key.split("|");


      links.push(
        {
          source,
          target,
          value
        }
      );

    }
  );


  const graph = {

    nodes:
      Array.from(
        nodesMap.values()
      ),

    links:
      links

  };


  // ==========================================================
  // DIMENSIONS
  // ==========================================================

  const width =
    Math.max(
      container.clientWidth,
      900
    );


  const regionCount =
    graph.nodes.filter(
      node =>
        node.type === "region"
    ).length;


  const height =
    Math.max(
      520,
      regionCount * 44
    );


  // ==========================================================
  // SVG
  // ==========================================================

  const svg =
    d3
      .select(
        container
      )
      .append(
        "svg"
      )
      .attr(
        "class",
        "direction-sankey-svg"
      )
      .attr(
        "viewBox",
        `0 0 ${width} ${height}`
      )
      .attr(
        "width",
        width
      )
      .attr(
        "height",
        height
      );


  // ==========================================================
  // SANKEY LAYOUT
  // ==========================================================

  const sankey =
    d3
      .sankey()
      .nodeId(
        node =>
          node.id
      )
      .nodeWidth(
        16
      )
      .nodePadding(
        14
      )
      .nodeAlign(
        d3.sankeyJustify
      )
      .extent(
        [
          [170, 20],
          [
            width - 190,
            height - 20
          ]
        ]
      );


  const sankeyGraph =
    sankey({
      nodes:
        graph.nodes.map(
          node => ({
            ...node
          })
        ),

      links:
        graph.links.map(
          link => ({
            ...link
          })
        )
    });


  // ==========================================================
  // LINK COLOUR
  // ==========================================================

  function getLinkColor(
    link
  ) {

    if (
      link.target.type
      === "prediction"
    ) {

      if (
        link.target.name
        === "Alzheimer's disease"
      ) {

        return "rgba(181, 87, 87, 0.38)";

      }


      if (
        link.target.name
        === "Normal aging"
      ) {

        return "rgba(67, 128, 105, 0.38)";

      }

    }


    return "rgba(80, 100, 150, 0.26)";

  }


  // ==========================================================
  // LINKS
  // ==========================================================

  svg
    .append(
      "g"
    )
    .attr(
      "fill",
      "none"
    )
    .selectAll(
      "path"
    )
    .data(
      sankeyGraph.links
    )
    .join(
      "path"
    )
    .attr(
      "class",
      "sankey-link"
    )
    .attr(
      "d",
      d3.sankeyLinkHorizontal()
    )
    .attr(
      "stroke",
      getLinkColor
    )
    .attr(
      "stroke-width",
      link =>
        Math.max(
          1,
          link.width
        )
    )
    .append(
      "title"
    )
    .text(
      link =>
        `${link.source.name} → ${link.target.name}
${link.value} subject${link.value === 1 ? "" : "s"}`
    );


  // ==========================================================
  // NODE COLOURS
  // ==========================================================

  function nodeColor(
    node
  ) {

    if (
      node.type === "region"
    ) {

      return "#64748b";

    }


    if (
      node.type === "direction"
    ) {

      return "#3157d5";

    }


    if (
      node.name
      === "Alzheimer's disease"
    ) {

      return "#b55757";

    }


    if (
      node.name
      === "Normal aging"
    ) {

      return "#438069";

    }


    return "#64748b";

  }


  // ==========================================================
  // NODES
  // ==========================================================

  const node =
    svg
      .append(
        "g"
      )
      .selectAll(
        "g"
      )
      .data(
        sankeyGraph.nodes
      )
      .join(
        "g"
      );


  node
    .append(
      "rect"
    )
    .attr(
      "x",
      d =>
        d.x0
    )
    .attr(
      "y",
      d =>
        d.y0
    )
    .attr(
      "height",
      d =>
        Math.max(
          1,
          d.y1 - d.y0
        )
    )
    .attr(
      "width",
      d =>
        d.x1 - d.x0
    )
    .attr(
      "rx",
      3
    )
    .attr(
      "fill",
      nodeColor
    )
    .append(
      "title"
    )
    .text(
      d =>
        `${d.name}
${Math.round(d.value)} subject${Math.round(d.value) === 1 ? "" : "s"}`
    );


  // ==========================================================
  // NODE LABELS
  // ==========================================================

  node
    .append(
      "text"
    )
    .attr(
      "class",
      "sankey-node-label"
    )
    .attr(
      "x",
      d => {

        if (
          d.type === "region"
        ) {

          return d.x0 - 8;

        }


        if (
          d.type === "prediction"
        ) {

          return d.x1 + 8;

        }


        return (
          (
            d.x0
            + d.x1
          )
          / 2
        );

      }
    )
    .attr(
      "y",
      d =>
        (
          d.y0
          + d.y1
        )
        / 2
    )
    .attr(
      "dy",
      "0.35em"
    )
    .attr(
      "text-anchor",
      d => {

        if (
          d.type === "region"
        ) {

          return "end";

        }


        if (
          d.type === "prediction"
        ) {

          return "start";

        }


        return "middle";

      }
    )
    .text(
      d =>
        d.name
    );

}


// ============================================================
// SUBJECT-LEVEL DIRECTIONALITY
// ============================================================

let subjectDirectionalityData = null;


const subjectMethodOrder = [
  "Zero-shot",
  "Random",
  "Cosine",
  "Euclidean",
  "Delta-kNN",
  "Proposed"
];


const subjectModelOrder = [
  "GPT-5.4",
  "Grok-4.20"
];


const directionSubjectSelect =
  document.getElementById(
    "direction-subject"
  );


// ============================================================
// LOAD SUBJECT DIRECTIONALITY DATA
// ============================================================

fetch(
  "data/subject_directionality.json"
)

  .then(response => {

    if (!response.ok) {

      throw new Error(
        `HTTP error: ${response.status}`
      );

    }

    return response.json();

  })

  .then(data => {

    subjectDirectionalityData =
      data;


    populateDirectionSubjects();

  })

  .catch(error => {

    console.error(
      "Could not load subject directionality data:",
      error
    );


    const container =
      document.getElementById(
        "subject-directionality-grid"
      );


    if (container) {

      container.innerHTML =
        "<p>Could not load subject-level directionality data.</p>";

    }

  });


// ============================================================
// SUBJECT SELECTOR
// ============================================================

function populateDirectionSubjects() {

  if (
    !subjectDirectionalityData
    || !directionSubjectSelect
  ) {

    return;

  }


  const subjects =
    Object.keys(
      subjectDirectionalityData.subjects
    );


  directionSubjectSelect.innerHTML =
    subjects
      .map(
        subject =>
          `
            <option value="${subject}">
              ${subject}
            </option>
          `
      )
      .join("");


  renderSubjectDirectionality(
    subjects[0]
  );

}


// ============================================================
// SUBJECT CHANGE
// ============================================================

if (directionSubjectSelect) {

  directionSubjectSelect.addEventListener(
    "change",
    () => {

      renderSubjectDirectionality(
        directionSubjectSelect.value
      );

    }
  );

}


// ============================================================
// GET ALL REGIONS MENTIONED FOR THIS SUBJECT
//
// GPT and Grok use exactly the same region columns.
// ============================================================

function getSubjectRegions(
  subjectData
) {

  const regions =
    new Set();


  Object.values(
    subjectData.models
  ).forEach(
    modelData => {


      Object.values(
        modelData.methods
      ).forEach(
        methodData => {


          Object.keys(
            methodData.regions
          ).forEach(
            region => {

              regions.add(
                region
              );

            }
          );

        }
      );

    }
  );


  return Array.from(
    regions
  ).sort();

}


// ============================================================
// SCORE CLASS
//
// +1 = towards AD
// -1 = towards CN
//  0 = neutral / unclear
// ============================================================

function getDirectionScoreClass(
  score
) {

  if (score === 1) {

    return "subject-direction-ad";

  }


  if (score === -1) {

    return "subject-direction-cn";

  }


  return "subject-direction-neutral";

}


// ============================================================
// SCORE LABEL
// ============================================================

function getDirectionScoreLabel(
  score
) {

  if (score === 1) {

    return "+1";

  }


  if (score === -1) {

    return "-1";

  }


  return "0";

}


// ============================================================
// PREDICTION SHORT LABEL
// ============================================================

function getPredictionShortLabel(
  predictionCode
) {

  if (
    predictionCode === "AD"
  ) {

    return "AD";

  }


  if (
    predictionCode === "CN"
  ) {

    return "CN";

  }


  return "—";

}


// ============================================================
// EVIDENCE SUMMARY
// ============================================================

function buildEvidenceSummary(
  methodData
) {

  const evidence =
    methodData.evidence;


  if (!evidence) {

    return "";

  }


  let majorityClass =
    "evidence-tie";


  let majoritySymbol =
    "—";


  if (
    evidence.majority
    === "Supports prediction"
  ) {

    majorityClass =
      "evidence-support";

    majoritySymbol =
      "✓";

  }


  else if (
    evidence.majority
    === "Opposes prediction"
  ) {

    majorityClass =
      "evidence-oppose";

    majoritySymbol =
      "✕";

  }


  return `
    <div
      class="evidence-summary ${majorityClass}"
      title="Supporting: ${evidence.supporting}
Opposing: ${evidence.opposing}
Neutral: ${evidence.neutral}"
    >

      <div class="evidence-ratio">

        <strong>
          ${evidence.supporting}
        </strong>

        <span>
          :
        </span>

        <strong>
          ${evidence.opposing}
        </strong>

        <span class="evidence-symbol">
          ${majoritySymbol}
        </span>

      </div>

      <div class="evidence-label">
        support : oppose
      </div>

    </div>
  `;

}


// ============================================================
// RENDER SUBJECT DIRECTIONALITY
// ============================================================

function renderSubjectDirectionality(
  subjectLabel
) {

  if (
    !subjectDirectionalityData
  ) {

    return;

  }


  const subjectData =
    subjectDirectionalityData
      .subjects?.[subjectLabel];


  if (!subjectData) {

    return;

  }


  const container =
    document.getElementById(
      "subject-directionality-grid"
    );


  const infoContainer =
    document.getElementById(
      "subject-info"
    );


  if (
    !container
    || !infoContainer
  ) {

    return;

  }


  // ==========================================================
  // SUBJECT INFORMATION
  // ==========================================================

  infoContainer.innerHTML = `

    <div class="subject-info-bar">

      <div>

        <span>
          Subject
        </span>

        <strong>
          ${subjectLabel}
        </strong>

      </div>


      <div>

        <span>
          True label
        </span>

        <strong>
          ${subjectData.true_label}
        </strong>

      </div>

    </div>
  `;


  // ==========================================================
  // SHARED REGION ORDER
  // ==========================================================

  const regions =
    getSubjectRegions(
      subjectData
    );


  let html = "";


  // ==========================================================
  // MODEL HEATMAPS
  // ==========================================================

  subjectModelOrder.forEach(
    model => {


      const modelData =
        subjectData.models?.[model];


      if (!modelData) {

        return;

      }


      const availableMethods =
        subjectMethodOrder.filter(
          method =>
            Object.prototype
              .hasOwnProperty
              .call(
                modelData.methods,
                method
              )
        );


      html += `

        <div class="subject-model-panel">

          <h5>
            ${model}
          </h5>


          <div class="subject-heatmap-scroll">


            <div
              class="subject-heatmap"
              style="
                grid-template-columns:
                  minmax(125px, 125px)
                  repeat(
                    ${regions.length},
                    minmax(48px, 48px)
                  )
                  minmax(90px, 90px)
                  minmax(105px, 105px);
              "
            >


              <div class="subject-heatmap-corner">
                Method
              </div>
      `;


      // ======================================================
      // REGION HEADERS
      // ======================================================

      regions.forEach(
        region => {

          html += `

            <div
              class="subject-region-heading"
              title="${region}"
            >

              <span>
                ${region}
              </span>

            </div>
          `;

        }
      );


      html += `

        <div class="subject-prediction-heading">
          Prediction
        </div>

        <div class="subject-evidence-heading">
          Evidence
        </div>
      `;


      // ======================================================
      // METHOD ROWS
      // ======================================================

      availableMethods.forEach(
        method => {


          const methodData =
            modelData.methods[
              method
            ];


          html += `

            <div class="subject-method-name">
              ${method}
            </div>
          `;


          // ==================================================
          // REGION VALUES
          // ==================================================

          regions.forEach(
            region => {


              const rawScore =
                methodData
                  .regions?.[region];


              // Region was not mentioned
              if (
                rawScore === undefined
              ) {

                html += `

                  <div
                    class="
                      subject-direction-cell
                      subject-direction-missing
                    "
                    title="${region}: not mentioned"
                  >
                    —
                  </div>
                `;

                return;

              }


              const score =
                Number(
                  rawScore
                );


              let directionDescription =
                "Neutral / unclear";


              if (score === 1) {

                directionDescription =
                  "Towards Alzheimer's disease";

              }


              else if (score === -1) {

                directionDescription =
                  "Towards normal aging";

              }


              html += `

                <div
                  class="
                    subject-direction-cell
                    ${getDirectionScoreClass(
                      score
                    )}
                  "
                  title="${region}: ${getDirectionScoreLabel(
                    score
                  )} — ${directionDescription}"
                >

                  ${getDirectionScoreLabel(
                    score
                  )}

                </div>
              `;

            }
          );


          // ==================================================
          // PREDICTION
          // ==================================================

          html += `

            <div
              class="
                subject-prediction-cell
                ${
                  methodData.prediction_code
                  === "AD"
                    ? "prediction-ad"
                    : "prediction-cn"
                }
              "
              title="${methodData.prediction}"
            >

              ${getPredictionShortLabel(
                methodData.prediction_code
              )}

            </div>
          `;


          // ==================================================
          // SUPPORT / OPPOSE
          // ==================================================

          html += `

            <div class="subject-evidence-cell">

              ${buildEvidenceSummary(
                methodData
              )}

            </div>
          `;

        }
      );


      html += `

          </div>

        </div>

      </div>
      `;

    }
  );


  container.innerHTML =
    html;

}