"use strict";
const $ = (id) => document.getElementById(id);
let token = "", selectedFile = null, previewURL = null, snapshot = null;
let predicting = false, loadingModel = false, pollTimer = null, elapsedTimer = null;
let modelListKey = "", loadedModel = "", loadedDevice = "", fileGeneration = 0;

function showError(message) {
  $("error-box").textContent = message || "";
  $("error-box").hidden = !message;
}

function updateButtons() {
  const busy = predicting || loadingModel || snapshot?.state === "loading" || snapshot?.state === "busy";
  $("predict-button").disabled = !selectedFile || snapshot?.state !== "ready" || busy;
  $("predict-label").textContent = predicting ? "사진을 분류하고 있어요…" : "이 사진 분류하기";
  $("drop-zone").disabled = predicting;
  $("change-photo").disabled = predicting;
  $("file-input").disabled = predicting;
  $("reload-model").disabled = !!busy;
  $("model-select").disabled = !!busy;
  $("device-select").disabled = !!busy;
}

function resetResult() {
  $("result-content").hidden = true;
  $("result-loading").hidden = true;
  $("result-empty").hidden = false;
}

function applyStatus(data) {
  snapshot = data;
  token = data.token;
  const titles = {ready: "모델 준비 완료", loading: "모델을 준비하고 있어요", busy: "사진을 분류하고 있어요", error: "모델을 확인해 주세요"};
  $("status-title").textContent = titles[data.state] || "모델 상태 확인";
  $("status-message").textContent = data.message;
  $("device-badge").textContent = data.state === "loading" ? "준비 중" : data.device === "xpu" ? "Intel GPU" : "CPU";
  $("device-badge").title = data.device_name || "";
  const nextKey = JSON.stringify(data.models);
  if (nextKey !== modelListKey) {
    const previous = $("model-select").value;
    $("model-select").replaceChildren();
    data.models.forEach((model, index) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = `${index === 0 ? "최근 저장 · " : ""}${model.label}`;
      $("model-select").append(option);
    });
    if (data.models.some((model) => model.id === previous)) $("model-select").value = previous;
    modelListKey = nextKey;
  }
  if (data.state === "ready") {
    if (data.model_id !== loadedModel || data.device !== loadedDevice) {
      $("model-select").value = data.model_id;
      $("device-select").value = data.device;
      loadedModel = data.model_id;
      loadedDevice = data.device;
    }
    $("epochs-value").textContent = `${data.epochs} 에포크`;
    $("image-size-value").textContent = `${data.image_size.join(" × ")} RGB`;
    $("model-summary").textContent = `${data.epochs} 에포크 · ${data.device === "xpu" ? "Intel GPU" : "CPU"}`;
    $("class-pills").replaceChildren(...data.class_names.map((name) => {
      const pill = document.createElement("span"); pill.textContent = name; return pill;
    }));
  }
  if (data.state === "error") {
    showError(data.message);
    $("model-details").open = true;
    $("model-summary").textContent = "설정을 확인하고 다시 불러와 주세요";
  }
  updateButtons();
}

async function refreshStatus() {
  clearTimeout(pollTimer);
  try {
    const response = await fetch("/api/status", {cache: "no-store"});
    if (!response.ok) throw new Error("서버 상태를 읽지 못했어요.");
    applyStatus(await response.json());
    if (snapshot.state === "loading" || snapshot.state === "busy") pollTimer = setTimeout(refreshStatus, 1200);
  } catch (error) {
    snapshot = null;
    $("status-title").textContent = "웹 실행 프로그램에 연결할 수 없어요";
    $("status-message").textContent = "START_IMAGE_WEB.cmd 실행 창이 열려 있는지 확인하고 페이지를 새로고침해 주세요.";
    $("device-badge").textContent = "연결 끊김";
    showError(error.message);
    updateButtons();
  }
}

async function request(path, body, contentType) {
  const response = await fetch(path, {
    method: "POST", headers: {"Content-Type": contentType, "X-App-Token": token}, body,
  });
  let result;
  try { result = await response.json(); }
  catch { throw new Error("서버 응답을 읽지 못했어요. 실행 창이 열려 있는지 확인해 주세요."); }
  if (!response.ok) throw new Error(result.error || "요청을 처리하지 못했어요.");
  return result;
}

async function chooseFile(file) {
  if (!file || predicting) return;
  showError("");
  const generation = ++fileGeneration;
  if (file.size === 0 || file.size > 20 * 1024 * 1024) {
    showError("비어 있지 않은 20MB 이하의 사진을 선택해 주세요."); return;
  }
  if (!/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) {
    showError("JPG, PNG, WEBP, BMP 사진 파일을 선택해 주세요."); return;
  }
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.src = url;
  try {
    await image.decode();
    if (image.naturalWidth * image.naturalHeight > 32_000_000) throw new Error("사진이 너무 커요. 3,200만 화소 이하로 줄여 주세요.");
    if (generation !== fileGeneration) { URL.revokeObjectURL(url); return; }
    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = url;
    selectedFile = file;
    $("preview-image").src = url;
    $("preview-image").alt = `${file.name} 미리보기`;
    $("preview-box").hidden = false;
    $("drop-zone").hidden = true;
    $("file-name").textContent = file.name;
    $("file-name").title = file.name;
    $("file-size").textContent = file.size >= 1024 * 1024 ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : `${Math.ceil(file.size / 1024)} KB`;
    resetResult();
    updateButtons();
  } catch (error) {
    URL.revokeObjectURL(url);
    if (generation === fileGeneration) showError(error.message.startsWith("사진") ? error.message : "이 사진을 열 수 없어요. 다른 JPG 또는 PNG 파일을 선택해 주세요.");
  }
}

function showResult(result, fileName) {
  $("predicted-name").textContent = result.predicted_name;
  $("result-file").textContent = fileName;
  $("probability-bars").replaceChildren();
  for (const item of result.probabilities) {
    const row = document.createElement("div");
    row.className = `probability-row${item.name === result.predicted_name ? " winner" : ""}`;
    const label = document.createElement("div"); label.className = "bar-label";
    const name = document.createElement("span"); name.textContent = item.name;
    const score = document.createElement("strong"); score.textContent = `${(item.value * 100).toFixed(1)}%`;
    label.append(name, score);
    const track = document.createElement("div"); track.className = "bar-track";
    track.setAttribute("role", "meter"); track.setAttribute("aria-label", `${item.name} 모델 점수`);
    track.setAttribute("aria-valuemin", "0"); track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", (item.value * 100).toFixed(1));
    const fill = document.createElement("div"); fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(100, item.value * 100))}%`;
    track.append(fill); row.append(label, track); $("probability-bars").append(row);
  }
  $("inference-time").textContent = `${result.seconds.toFixed(2)}초`;
  $("inference-device").textContent = result.device.startsWith("xpu") ? "Intel GPU 사용" : "CPU 사용";
  $("result-empty").hidden = true;
  $("result-loading").hidden = true;
  $("result-content").hidden = false;
}

$("drop-zone").addEventListener("click", () => $("file-input").click());
$("change-photo").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) => {
  chooseFile(event.target.files[0]); event.target.value = "";
});
for (const eventName of ["dragover", "drop"]) document.addEventListener(eventName, (event) => event.preventDefault());
$("input-title").closest("section").addEventListener("dragover", () => $("drop-zone").classList.add("drag-over"));
$("input-title").closest("section").addEventListener("dragleave", () => $("drop-zone").classList.remove("drag-over"));
$("input-title").closest("section").addEventListener("drop", (event) => {
  $("drop-zone").classList.remove("drag-over");
  if (event.dataTransfer.files.length > 1) { showError("사진은 한 번에 한 장씩 선택해 주세요."); return; }
  chooseFile(event.dataTransfer.files[0]);
});

$("predict-button").addEventListener("click", async () => {
  if (!selectedFile || predicting || snapshot?.state !== "ready") return;
  const file = selectedFile;
  predicting = true;
  showError(""); updateButtons();
  $("result-empty").hidden = true; $("result-content").hidden = true; $("result-loading").hidden = false;
  $("result-panel").setAttribute("aria-busy", "true");
  $("loading-message").textContent = "첫 번째 분류는 GPU 준비로 시간이 더 걸릴 수 있어요.";
  const started = performance.now();
  $("elapsed-time").textContent = "0.0초";
  elapsedTimer = setInterval(() => {
    const seconds = (performance.now() - started) / 1000;
    $("elapsed-time").textContent = `${seconds.toFixed(1)}초`;
    if (seconds > 45) $("loading-message").textContent = "아직 계산 중이에요. 계속 지연되면 실행 창의 오류 메시지를 확인해 주세요.";
  }, 100);
  try {
    const result = await request("/api/predict", file, "application/octet-stream");
    showResult(result, file.name);
  } catch (error) {
    resetResult(); showError(error.message);
  } finally {
    clearInterval(elapsedTimer);
    predicting = false;
    $("result-panel").setAttribute("aria-busy", "false");
    await refreshStatus();
  }
});

$("reload-model").addEventListener("click", async () => {
  if (!token) { await refreshStatus(); return; }
  loadingModel = true; showError(""); resetResult(); updateButtons();
  try {
    await request("/api/model", JSON.stringify({model_id: $("model-select").value, device: $("device-select").value}), "application/json");
  } catch (error) { showError(error.message); }
  finally { loadingModel = false; await refreshStatus(); }
});

if (location.protocol === "file:") {
  showError("START_IMAGE_WEB.cmd를 더블클릭해서 열어 주세요. GPU 분류에는 Python 프로그램 연결이 필요해요.");
  $("status-title").textContent = "실행 파일로 열어 주세요";
  $("status-message").textContent = "프로젝트 폴더의 START_IMAGE_WEB.cmd를 실행하면 자동으로 연결돼요.";
} else {
  refreshStatus();
}
