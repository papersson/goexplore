import * as d3 from 'd3'
import {useRef} from 'react'

const Cell = () => {
  const svgRef = useRef();

  function gridData(cell, size, n) {
    let data = new Array();
    let x = 0;
    let y = 0;
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

  const size = 160;
  const n = 8;
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

  var data = gridData(cell, size, n);

  const svg = d3.select(svgRef.current);

  const grid = svg
    .append("svg")
    .attr("width", size)
    .attr("height", size)
    .attr("style", "border: solid 1px black");

  const row = grid
    .selectAll(".row")
    .data(data)
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

  return <svg ref={svgRef} height={size/n} width={size/n}></svg>
}

export default Cell;
