# Neural Style Transfer
<p>This project is all about Art generation using Neural Style Transfer by <a href="https://arxiv.org/abs/1508.06576">Gatys et al</a>. In this project, neural style transfer algorithm is implemented. Using it, we generate novel artistic images. Most of the algorithms optimize a cost function to get a set of parameter values. In Neural Style Transfer, we'll optimize a cost function to get pixel values!</p><br/>

<p>Neural Style Transfer (NST) is one of the most fun techniques in deep learning. As seen below, it merges two images, namely, a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S.</p><br/>

<img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/images/content_plus_style.png" /><br/>

<p>We'll use the concept of transfer learning, i.e. we will be using a pre-built neural network called <a href="http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html">VGG network</a>. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the earlier layers) and high level features (at the deeper layers). The weights for this pre-built network has been taken from <a href="http://www.vlfeat.org/matconvnet/pretrained/">here</a> and <a href="http://www.robots.ox.ac.uk/~vgg/research/very_deep/">this</a> is the corresponding website. This project is implemented as part of the <a href="https://github.com/adityachandupatla/ML_Coursera">deeplearning specialization</a> from Coursera.</p><br/>
<h2>Running the Project</h2>
<ul>
  <li>Make sure you have Python3 installed</li>
  <li>Clone the project to your local machine and open it in one of your favorite IDE's which supports Python code</li>
  <li>Make sure you have the following dependencies installed:
    <ol>
      <li><a href="http://www.numpy.org/">Numpy</a></li>
      <li><a href="https://www.tensorflow.org/">Tensorflow</a></li>
      <li><a href="https://pillow.readthedocs.io/">Pillow</a></li>
      <li><a href="https://www.scipy.org/">Scipy</a></li>
      <li><a href="https://matplotlib.org/">Matplotlib</a></li>
    </ol>
  </li>
  <li>Please download the pre-built VGG network weights from the previously specified link and place it in this project's pretrained-model folder. The size of the weights of this network is humongous, close to 550MB, and is therefore not checked into Git. If you are unable to find the dataset, please do let me know, so that I can help you find out.</li>
  <li>Pick 2 images, each of size 800 X 600 pixels. One image is the content image, and the other is the style image. Put both these images in the image folder and specify their path in the code. See line 162 and 164.</li>
  <li>Run nst.py</li>
</ul>
If you find any problem deploying the project in your machine, please do let me know.

<h2>Technical Skills</h2>
This project is developed to showcase my following programming abilities:
<ul>
  <li>Python</li>
  <li>Computer Vision based applications</li>
  <li>Transfer learning - Building on top of existing networks such as the VGG network</li>
  <li>Use of high level programming frameworks such as Tensorflow</li>
</ul>

<h2>Development</h2>
<ul>
  <li>Sublimt Text has been used to program the application. No IDE has been used.</li>
  <li>Command line has been used to interact with the application.</li>
  <li>The project has been tested on Python3 version: 3.6.1.</li>
</ul>

<h2>Working</h2>
<p>Architectural Diagram:<br/><img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/images/NST.png" /></p>
<p>We will build the NST algorithm in three steps:
  <ul>
    <li>Build the content cost function <br/><img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/images/content_cost_function.png" /></li>
    <li>Build the style cost function <br/><img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/images/style_cost_function.png" /></li>
    <li>Put them together <br/><img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/images/combined_cost_function.png" /></li>
  </ul>
Here,  nH,nW  and  nC  are the height, width and number of channels of the hidden layer we have chosen, and appear in a normalization term in the cost.
</p><br/>

<p>The earlier (shallower) layers of a ConvNet tend to detect lower-level features such as edges and simple textures, and the later (deeper) layers tend to detect higher-level features such as more complex textures as well as object classes.<br/><br/><img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/images/hidden_layers.png" /><br/><br/>We would like the "generated" image G to have similar content as the input image C. Let us pick one particular hidden layer to use. Now, we'll set the image C as the input to the pretrained VGG network, and run forward propagation. Let  a(C)  be the hidden layer activations in the layer we had chosen. This will be a  nH×nW×nC  tensor. Repeat this process with the image G: Set G as the input, and run forward progation. Let a(G) be the corresponding hidden layer activation.</p><br/>
<p>The style matrix is also called a "Gram matrix." In linear algebra, the Gram matrix G of a set of vectors  (v1,…,vn)  is the matrix of dot products, whose entries are  Gij = np.dot(vi,vj) . In other words,  Gij  compares how similar  vi  is to  vj : If they are highly similar, we would expect them to have a large dot product, and thus for Gij to be large.<br/><img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/images/gram.png" />
</p><br/>

<p>The generated images are saved in the output folder. You can see the following example for inspiration:<br/>

<h3>Content image</h3> <img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/images/my_goa_image.jpg"  width="400" height="300" />(This is an image of mine, from Goa trip!)
<h3>Style image</h3> <p>(Taken from <a href="https://fineartamerica.com/featured/2-art-abstract-work-odon-czintos.html">here</a>)</p><img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/images/abstract-work-odon-czintos.jpg" width="400" height="300" />
<h3>Generated image</h3> <img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/output/generated_image.jpg" width="400" height="300" />
<h3>Intermediate images</h3><p>This is what the algorithm does during the intermediate steps. It took close to close 1 hour for running 200 iterations in order to generate the final image!</p><br/>

<p float="left">
  <img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/output/0.png" height="225" width="300" />
  <img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/output/20.png" height="225" width="300" />
  <img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/output/60.png" height="225" width="300" />
  <img src="https://github.com/adityachandupatla/neural_style_transfer/blob/master/output/180.png" height="225" width="300" />
</p><br/>
</p>
<h2>TO DO</h2>
<ul>
  <li>I plan on integrating this application with Android to develop a mobile application, similar to <a href="https://prisma-ai.com/">Prisma</a>!</li>
</ul>
<br/><br/>
Use this, report bugs, raise issues and Have fun. Do whatever you want! I would love to hear your feedback :)

~ Happy Coding
