let fileNames = null;
let speed = 0.5;
let flag = true;
let input = null;
let i = 0;

fetch("./data.json")
  .then((response) => response.json())
  .then((data) => {
    fileNames = data["file_names"];
    draw(fileNames);
  });

let draw = (fileNames) => {
  console.log(flag);
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
  ctx = canvas.getContext("2d");
  img.setAttribute("width", "320");
  img.setAttribute("height", "420");

  let switchImage = () => {
    img.setAttribute("src", fileNames[i++]);
    img.onload = () => {
      if (i >= fileNames.length) i = fileNames.length - 1;
      ctx.drawImage(img, 0, 0, 320, 420);
      ctx.font = "24px serif";
      ctx.textAlign = "center";
      ctx.fillText("Frame: " + i, 160, 412);
    };
    if (flag) {
      setTimeout(switchImage, (2 * speed * 1000) / 30);
    }
  };
  switchImage();
};

let toggle = (b) => {
  flag = !flag;
  if (flag) {
    draw(fileNames);
  }
};
