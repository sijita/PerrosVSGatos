import React from "react";
import Banner from "./images/banner.jpg";
import * as tf from "@tensorflow/tfjs";
import { useState, useEffect } from "react";


export default function Main() {
    const [model, setModel] = useState()

    const loadModel = async () => {
        try {
            const model = await tf.loadLayersModel("/model.json");
            setModel(model);
            console.log("Modelo cargado");
        } catch (error) {
            console.log(error)
        }
    }

    useEffect(() => {
        loadModel()
    }, [])

    useEffect(() => {
      var tamano = 400;
      var video = document.getElementById("video");
      var canvas = document.getElementById("canvas");
      var otrocanvas = document.getElementById("otrocanvas");
      var ctx = canvas.getContext("2d");
      var currentStream = null;
      var facingMode = "user";

    window.onload = function() {
      mostrarCamara();
    }

    predecir();

    function mostrarCamara() {
      var opciones = {
        audio: false,
        video: {
          width: tamano, height: tamano
        }
      }
      
      if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia(opciones)
            .then(function(stream) {
              currentStream = stream;
              video.srcObject = currentStream;
              procesarCamara();
              predecir();
            })
            .catch(function(err) {
              alert("No se pudo utilizar la camara :(");
              console.log(err);
              alert(err);
            })
      } else {
        alert("No existe la funcion getUserMedia");
      }
    }

    function cambiarCamara() {
          if (currentStream) {
              currentStream.getTracks().forEach(track => {
                  track.stop();
              });
          }

          facingMode = facingMode === "user" ? "environment" : "user";

          var opciones = {
              audio: false,
              video: {
                  facingMode: facingMode, width: tamano, height: tamano
              }
          };


          navigator.mediaDevices.getUserMedia(opciones)
              .then(function(stream) {
                  currentStream = stream;
                  video.srcObject = currentStream;
              })
              .catch(function(err) {
                  console.log("Oops, hubo un error", err);
              })
      }

    function procesarCamara() {
      ctx.drawImage(video, 0, 0, tamano, tamano, 0, 0, tamano, tamano);
      setTimeout(procesarCamara, 20);
    }

    function predecir() {
      if (model != null) {
        resample_single(canvas, 100, 100, otrocanvas);

        //Hacer la predicción
        var ctx2 = otrocanvas.getContext("2d");
        var imgData = ctx2.getImageData(0,0, 100, 100);

        var arr = [];
        var arr100 = [];

        for (var p=0; p < imgData.data.length; p+= 4) {
          var rojo = imgData.data[p] / 255;
          var verde = imgData.data[p+1] / 255;
          var azul = imgData.data[p+2] / 255;

          var gris = (rojo+verde+azul)/3;

          arr100.push([gris]);
          if (arr100.length === 100) {
            arr.push(arr100);
            arr100 = [];
          }
        }

        arr = [arr];

        var tensor = tf.tensor4d(arr);
        var resultado = model.predict(tensor).dataSync();

        var respuesta;
        if (resultado <= .5) {
          respuesta = "Gato";
        } else {
          respuesta = "Perro";
        }
        document.getElementById("resultado").innerHTML = respuesta;

      }

      setTimeout(predecir, 150);
    }

    /**
       * Hermite resize - fast image resize/resample using Hermite filter. 1 cpu version!
       * 
       * @param {HtmlElement} canvas
       * @param {int} width
       * @param {int} height
       * @param {boolean} resize_canvas if true, canvas will be resized. Optional.
       * Cambiado por RT, resize canvas ahora es donde se pone el chiqitillllllo
       */
      function resample_single(canvas, width, height, resize_canvas) {
          var width_source = canvas.width;
          var height_source = canvas.height;
          width = Math.round(width);
          height = Math.round(height);

          var ratio_w = width_source / width;
          var ratio_h = height_source / height;
          var ratio_w_half = Math.ceil(ratio_w / 2);
          var ratio_h_half = Math.ceil(ratio_h / 2);

          var ctx = canvas.getContext("2d");
          var ctx2 = resize_canvas.getContext("2d");
          var img = ctx.getImageData(0, 0, width_source, height_source);
          var img2 = ctx2.createImageData(width, height);
          var data = img.data;
          var data2 = img2.data;

          for (var j = 0; j < height; j++) {
              for (var i = 0; i < width; i++) {
                  var x2 = (i + j * width) * 4;
                  var weight = 0;
                  var weights = 0;
                  var weights_alpha = 0;
                  var gx_r = 0;
                  var gx_g = 0;
                  var gx_b = 0;
                  var gx_a = 0;
                  var center_y = (j + 0.5) * ratio_h;
                  var yy_start = Math.floor(j * ratio_h);
                  var yy_stop = Math.ceil((j + 1) * ratio_h);
                  for (var yy = yy_start; yy < yy_stop; yy++) {
                      var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
                      var center_x = (i + 0.5) * ratio_w;
                      var w0 = dy * dy; //pre-calc part of w
                      var xx_start = Math.floor(i * ratio_w);
                      var xx_stop = Math.ceil((i + 1) * ratio_w);
                      for (var xx = xx_start; xx < xx_stop; xx++) {
                          var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
                          var w = Math.sqrt(w0 + dx * dx);
                          if (w >= 1) {
                              //pixel too far
                              continue;
                          }
                          //hermite filter
                          weight = 2 * w * w * w - 3 * w * w + 1;
                          var pos_x = 4 * (xx + yy * width_source);
                          //alpha
                          gx_a += weight * data[pos_x + 3];
                          weights_alpha += weight;
                          //colors
                          if (data[pos_x + 3] < 255)
                              weight = weight * data[pos_x + 3] / 250;
                          gx_r += weight * data[pos_x];
                          gx_g += weight * data[pos_x + 1];
                          gx_b += weight * data[pos_x + 2];
                          weights += weight;
                      }
                  }
                  data2[x2] = gx_r / weights;
                  data2[x2 + 1] = gx_g / weights;
                  data2[x2 + 2] = gx_b / weights;
                  data2[x2 + 3] = gx_a / weights_alpha;
              }
          }


          ctx2.putImageData(img2, 0, 0);
      }

    })

  return (
      
    <>
      <h1 className="text-center p-5 fw-bold">Perros Vs Gatos</h1>
      <div className="px-4 py-2 my-2 text-center border-bottom">
        <img
          className="d-block mx-auto mb-2 img-fluid"
          src={Banner}
          alt="perrosvsgatos"
          width="300"
          height="200"
        />
        <div className="col-lg-6 mx-auto">
          <p className="lead mb-0">
            Clasificación de imágenes (Perro o Gato) usando la cámara web con
            ayuda de Tensorflow.js
          </p>
        </div>
      </div>
      <div className="container mt-5 mb-5">
        <div className="row">
          <div className="col-12 col-md-4 offset-md-4 text-center">
            <video id="video" playsInline autoPlay style={{ width: "1px" }}></video>
            <button className="btn btn-primary mb-2 d-none" id="cambiar-camara">
              Cambiar camara
            </button>
            <canvas
              id="canvas"
              width="400"
              height="400"
              style={{ maxWidth: "100%" }}
            ></canvas>
            <canvas
              id="otrocanvas"
              width="150"
              height="150"
              style={{ display: "none" }}
            ></canvas>
            <div id="resultado" className="fw-bold fs-1"></div>
          </div>
        </div>
      </div>
    </>
  );
}
