function gridData(cell, size) {
  let data = new Array();
  let x = 0;
  let y = 0;
  const n = 8;
  let width = size / n;
  let height = size / n;

  for (let i = 0; i < n; i++) {
    data.push(new Array());

    for (let j = 0; j < n; j++) {
      data[i].push({
        x: x,
        y: y,
        width: width,
        height: height,
        color: cell[i][j] == 1 ? "black" : "white",
      });
      x += width;
    }
    x = 1;
    y += height;
  }
  return data;
}

const archive = d3
  .select("#archive")
  .attr(
    "style",
    "border-collapse: collapse; border: solid 1px black; margin-top: 20px;"
  );

archive.append("thead").append("tr");

const columnNames = ["Cell", "Score", "Visits"];
const headers = archive
  .select("tr")
  .selectAll("th")
  .data(columnNames)
  .enter()
  .append("th")
  .text((d) => d)
  .attr("style", "background-color: #aaa");

const rowData = [
  { id: 1, name: "bob" },
  { id: 2, name: "alice" },
  { id: 3, name: "claus" },
  { id: 4, name: "dave" },
];
const archiveRows = archive
  .append("tbody")
  .selectAll("tr")
  .data(rowData, (d) => d.id)
  .enter()
  .append("tr"); // IDK KEK

const size = 160;
const cell = [
  [1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 1, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 1],
];

var gridData = gridData(cell, size);

const grid = d3
  .select("#grid")
  .append("svg")
  .attr("width", size)
  .attr("height", size)
  .attr("style", "border: solid 1px black");

const row = grid
  .selectAll(".row")
  .data(gridData)
  .enter()
  .append("g")
  .attr("class", "row");

const column = row
  .selectAll(".square")
  .data((d) => d)
  .enter()
  .append("rect")
  .attr("class", "square")
  .attr("x", (d) => d.x)
  .attr("y", (d) => d.y)
  .attr("width", (d) => d.width)
  .attr("height", (d) => d.height)
  .attr("fill", (d) => d.color)
  .attr("stroke", (d) => (d.color == "black" ? "black" : "white"));
