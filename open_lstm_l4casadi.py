import torch
import torch.nn as nn

class OpenLSTML4casadi(nn.Module):
    def __init__(self, n_context, n_inputs, batch_size, sequence_length, is_estimator=True):
        super(OpenLSTML4casadi, self).__init__()
        self.n_context = n_context  # Number of steps for context
        self.n_inputs = n_inputs  # Input size (num features)
        self.sequence_length=sequence_length
        self.is_estimator = is_estimator
        self.batch_size = batch_size
        # Define the LSTM model with projection
        self.model = nn.LSTM(input_size=n_inputs, hidden_size=128, proj_size=1, num_layers=1, batch_first=False)
        self.hn = None
        self.cn = None

        # Apply custom initialization to LSTM parameters
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # Initialize weights with Gaussian distribution
                nn.init.normal_(param, mean=0.0, std=1e-4)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.constant_(param, 0)

    def forward(self, u_train):
        # Detect the input shape and reshape if necessary
        if (len(u_train.shape)) == 2:
        # Reshape from [batch_size * sequence_length, input_size] to [sequence_length, batch_size, input_size]

            #batch_size_sequence_length = u_train.shape[0]

            #if batch_size_sequence_length % self.sequence_length == 0:
                #batch_size = batch_size_sequence_length // self.sequence_length
            #else:
                #raise ValueError("The total size is not divisible by the sequence length.")

            # Now reshape after verifying
            u_train = u_train.view(self.sequence_length, self.batch_size, -1)  # Shape: [sequence_length, batch_size, input_size]

        #print(u_train.shape)
        if self.is_estimator:
            y1 = self.estimate_state(u_train[:, :, :self.n_inputs],
                                     u_train[:, :, self.n_inputs:], self.n_context)
            y2 = self.predict_state(u_train[:, :, :self.n_inputs], self.n_context)

            y_sim = torch.cat((y1, y2), dim=0)
        else:
            state = (self.hn, self.cn)
            y_sim, _ = self.model(u_train, state)
            #y_sim = y_sim.view(-1, self.n_inputs) # TO Flatten the output to [batch_size * sequence_length, num_features]
        return y_sim

    def estimate_state(self, u_train, y_train, nstep):
        # Detect the input shape and reshape if necessary
        if len(u_train.shape) == 2:
            #batch_size_sequence_length = u_train.shape[0]
            #input_size = u_train.shape[1]

            # Calculate sequence_length and batch_size assuming sequence_length is known
            #batch_size = batch_size_sequence_length // self.sequence_length
            
            u_train = u_train.view(self.sequence_length, self.batch_size, self.n_inputs)

        y_est = []
        # Initialize hidden and cell states
        hn = torch.zeros(1, u_train.size(1), 1, requires_grad=False)  # proj_size is 1 #
        cn = torch.zeros(1, u_train.size(1), 128, requires_grad=False)  # hidden_size is 128 #

        for i in range(nstep):
            # Feed in the known output to estimate state
            out, (hn, cn) = self.model(u_train[i, :, :].unsqueeze(1),
                                       (y_train[i, :, :].view(hn.shape), cn))
            y_est.append(out)

        y_sim = torch.cat(y_est, dim=0) #concat over sequence_lenght dimension
        #y_sim = y_sim.view(-1, self.n_inputs) # TO Flatten the output to [batch_size * sequence_length, num_features]
        self.hn, self.cn = (hn, cn)  # Store hidden and cell states for prediction
        return y_sim

    def predict_state(self, u_train, nstep):
        # Detect the input shape and reshape if necessary
        if len(u_train.shape) == 2:
            # Reshape from [batch_size * sequence_length, input_size] to [batch_size, sequence_length, input_size]
            #batch_size_sequence_length = u_train.shape[0]
            #input_size = u_train.shape[1]

            # Calculate sequence_length and batch_size assuming sequence_length is known
            #batch_size = batch_size_sequence_length // self.sequence_length
            
            u_train = u_train.view(self.sequence_length, self.batch_size, self.n_inputs)
        # Predict using the stored hidden state
        state = (self.hn, self.cn)
        y_sim, _ = self.model(u_train[nstep:, :, :], state)
        #y_sim = y_sim.view(-1, self.n_inputs) # TO Flatten the output to [batch_size * sequence_length, num_features]
        return y_sim

    def get_model(self):
        return self.model
