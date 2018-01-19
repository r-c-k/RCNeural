class RCNeural {

  RCMatrixUtil util = new RCMatrixUtil();

  private int inputNodes;
  private int hiddenNodes;
  private int outputNodes;
  private float learningrate;

  private float[][] weights_inputHidden;
  private float[][] weights_hiddenOutput;

  RCNeural (int n_inputNodes, int n_hiddenNodes, int n_outputNodes, float learningrate) {

    this.inputNodes = n_inputNodes;
    this.hiddenNodes = n_hiddenNodes;
    this.outputNodes = n_outputNodes;
    this.learningrate = learningrate;

    this.weights_inputHidden = util.sub(util.rand(this.hiddenNodes, this.inputNodes), 0.5);
    this.weights_hiddenOutput = util.sub(util.rand(this.outputNodes, this.hiddenNodes), 0.5);

  }

  public void train (float[] inputs_list, float[] targets_list) {

    float[][] inputs = util.make2d(inputs_list);
    float[][] targets = util.make2d(targets_list);

    float[][] hidden_inputs = util.dot(this.weights_inputHidden, inputs);
    float[][] hidden_outputs = this.activation(hidden_inputs);

    float[][] final_inputs = util.dot(this.weights_hiddenOutput, hidden_outputs);
    float[][] final_outputs = this.activation(final_inputs);

    float[][] output_errors = util.sub(targets, final_outputs);
    float[][] hidden_errors = util.dot(util.transpose(this.weights_hiddenOutput), output_errors);

    this.weights_hiddenOutput = util.add(this.weights_hiddenOutput, util.mult(util.dot((util.mult(util.mult(output_errors, final_outputs), util.sub(1.0, final_outputs))), util.transpose(hidden_outputs)), this.learningrate));
    this.weights_inputHidden = util.add(this.weights_inputHidden, util.mult(util.dot((util.mult(util.mult(hidden_errors, hidden_outputs), util.sub(1.0, hidden_outputs))), util.transpose(inputs)), this.learningrate));
    
  }

  public float[][] query (float[] inputs_list) {
    
    float[][] inputs = util.make2d(inputs_list);
    
    float[][] hidden_inputs = util.dot(this.weights_inputHidden, inputs);
    float[][] hidden_outputs = this.activation(hidden_inputs);
    
    float[][] final_inputs = util.dot(this.weights_hiddenOutput, hidden_outputs);
    float[][] final_outputs = this.activation(final_inputs);
    
    return final_outputs;
  }

  // ######################################################
  // ##                     ACTIVATION                   ##
  // ######################################################

  private float[][] activation (float[][] matrix) {

    int rows = util.getRows(matrix);
    int cols = util.getCols(matrix);

    float[][] resultMatrix = new float[rows][cols];

    for (int c = 0; c < cols; c++) {
      for (int r = 0; r < rows; r++) {
        resultMatrix[r][c] = 1.0 / (1.0 + (pow(exp(1), -matrix[r][c])));
      }
    }

    return resultMatrix;
  }
};
