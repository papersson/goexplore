let fileNames = null;
let archiveData = null;
let speed = 0.5;
let flag = false;
let input = null;
let i = 0;

fetch("./data.json")
  .then((response) => response.json())
  .then((data) => {
    console.log(data);
    fileNames = data["file_names"];
    archiveData = data["archive"];
    draw(fileNames, archiveData);
  });

let draw = (fileNames, archiveData) => {
  //fileNames = data["file_names"];
  //archiveData = data["archive"];

  let slider = document.getElementById("slider");
  slider.oninput = () => {
    speed = slider.value;
  };

  input = document.getElementById("frameInput");
  input.addEventListener("keydown", (event) => {
    const keyName = event.key;
    if (keyName === "Enter") {
      i = input.value;
    }
  });
  let canvas = null;
  let ctx = null;
  let img = document.createElement("img");
  canvas = document.getElementById("canvas");
  cellCanvas = document.getElementById("cellCanvas");
  ctx = canvas.getContext("2d");
  img.setAttribute("width", "320");
  img.setAttribute("height", "420");

  updateRow(archiveData[i]);

  let switchImage = () => {
    let [cell, seen, visits, update] = archiveData[i];

    img.setAttribute("src", fileNames[i++]);
    img.onload = () => {
      if (i >= fileNames.length) i = fileNames.length - 1;
      ctx.drawImage(img, 0, 0, 320, 420);
      ctx.font = "24px serif";
      ctx.textAlign = "center";
      ctx.fillText("Frame: " + i, 160, 412);
      cellCanvas.innerHTML = "";
      let currentCell = drawCell(cell);
      currentCell.className = "bigCell";
      cellCanvas.append(currentCell);
    };

    //console.log(update);
    //if (update.length !== 0) {
    updateRow(archiveData[i]);
    //}

    if (flag) {
      setTimeout(switchImage, (2 * speed * 1000) / 30);
    }
  };
  switchImage();
};

const updateRow = (archiveFrame) => {
  let [currCell, seen, visits, update] = archiveFrame;
  let [score, trajectory] = update;

  // Add new row if not in archive
  if (update.length !== 0) {
    //&& update.length !== 0
    // New cell in archive

    let archive = document.getElementById("archive");
    let newRow = archive.insertRow();

    let cellCell = newRow.insertCell();
    smallCell = drawCell(currCell);
    smallCell.className = "smallCell";
    cellCell.appendChild(smallCell);

    let scoreCell = newRow.insertCell();
    let scoreText = document.createTextNode(0 | score);
    scoreCell.appendChild(scoreText);

    let visitsCell = newRow.insertCell();
    let visitsText = document.createTextNode(visits);
    visitsCell.appendChild(visitsText);
  }

  // Update archive cell
  let cells = document.getElementsByClassName("smallCell");
  console.log(cells);
  let highlight = true;
  //while (!highlight && n < cells.length) {
  for (let k = 0; k < cells.length; k++) {
    let cell = cells[k];
    let tds = cell.getElementsByTagName("td");
    for (let j = 0; j < tds.length; j++) {
      //      console.log("currcell: " + currCell.flat()[j]);
      //     console.log("td: " + tds[j].innerText);
      if (Number(tds[j].innerText) !== currCell.flat()[j]) {
        highlight = false;
      }
    }
    if (highlight) {
      tr = cell.parentNode.parentNode;
      console.log(cell);
      console.log(tr);
      tr.style.backgroundColor = "yellow";
      tds = tr.childNodes;
      //tds[1] = score; // Score
      tds[2] = visits; // Visits
      break;
    }
  }
};

let drawCell = (cell) => {
  //const body = document.body;
  const table = document.createElement("table");
  table.className = "cell";

  for (let i = 0; i < cell.length; i++) {
    const tr = table.insertRow();

    for (let j = 0; j < cell[i].length; j++) {
      const color = cell[i][j] === 0 ? "red" : "black";
      const td = tr.insertCell();
      td.appendChild(document.createTextNode(""));
      //td.style.border = "1px solid " + color;
      td.className = cell[i][j] === 255 ? "whitePixel" : "blackPixel";
      td.innerText = cell[i][j];
    }
  }
  return table;
};

let toggle = (b) => {
  flag = !flag;
  if (flag) {
    draw(fileNames, archiveData);
    b.innerHTML = "Pause";
  } else {
    b.innerHTML = "Play";
  }
};
