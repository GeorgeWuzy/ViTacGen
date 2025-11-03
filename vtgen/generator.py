import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torchvision.models as models  

class ResBlock(nn.Module):  
    def __init__(self, dim, norm_layer):  
        super().__init__()  
        
        self.conv_block = nn.Sequential(  
            nn.ReflectionPad2d(1),  
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),  
            norm_layer(dim),  
            nn.ReLU(True),  
            nn.ReflectionPad2d(1),  
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),  
            norm_layer(dim)  
        )  
        
    def forward(self, x):  
        return x + self.conv_block(x)

class CrossModalAttention(nn.Module):  
    def __init__(self, query_dim, key_dim, num_heads=8):  
        super().__init__()  
        self.num_heads = num_heads  
        self.scale = (key_dim // num_heads) ** -0.5  
        
        self.to_q = nn.Linear(query_dim, key_dim)  
        self.to_k = nn.Linear(key_dim, key_dim)  
        self.to_v = nn.Linear(key_dim, key_dim)  
        self.to_out = nn.Linear(key_dim, query_dim)  
        
    def forward(self, query, key_value):  
        B, H, W = query.shape[0], query.shape[2], query.shape[3]  
        
        query = query.flatten(2).transpose(1, 2)  # B, HW, C  
        key_value = key_value.flatten(2).transpose(1, 2)  # B, HW, C  
        
        q = self.to_q(query)  
        k = self.to_k(key_value)  
        v = self.to_v(key_value)  
        
        q = q.view(B, -1, self.num_heads, q.shape[-1] // self.num_heads).transpose(1, 2)  
        k = k.view(B, -1, self.num_heads, k.shape[-1] // self.num_heads).transpose(1, 2)  
        v = v.view(B, -1, self.num_heads, v.shape[-1] // self.num_heads).transpose(1, 2)  
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)  
        
        out = torch.matmul(attn, v)  
        out = out.transpose(1, 2).reshape(B, -1, q.shape[-1] * self.num_heads)  
        out = self.to_out(out)  
        
        out = out.transpose(1, 2).view(B, -1, H, W)  
        return out  

class Encoder(nn.Module):  
    def __init__(self, input_channels, dim, n_downsamplings, max_dim=2048, norm_layer=nn.InstanceNorm2d):  
        super().__init__()  
        
        # Initial convolution  
        layers = [  
            nn.ReflectionPad2d(3),  
            nn.Conv2d(input_channels, dim, kernel_size=7, padding=0, bias=False),  
            norm_layer(dim),  
            nn.ReLU(True)  
        ]  
        
        # Downsampling layers  
        curr_dim = dim  
        for i in range(n_downsamplings):  
            next_dim = min(curr_dim * 2, max_dim)
            layers.extend([  
                nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1, bias=False),  
                norm_layer(next_dim),  
                nn.ReLU(True)  
            ])  
            curr_dim = next_dim  
            
        self.model = nn.Sequential(*layers)  
        self.out_channels = curr_dim  
        
    def forward(self, x):  
        return self.model(x)  

class Decoder(nn.Module):  
    def __init__(self, input_channels, output_channels, n_upsamplings, norm_layer=nn.InstanceNorm2d):  
        super().__init__()  
        
        curr_dim = input_channels
        layers = []  
        
        # Upsampling layers  
        for i in range(n_upsamplings):  
            next_dim = curr_dim // 2
            layers.extend([  
                nn.ConvTranspose2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1, bias=False),  
                norm_layer(next_dim),  
                nn.ReLU(True)  
            ])  
            curr_dim = next_dim  
            
        # Output layer  
        layers.extend([  
            nn.ReflectionPad2d(3),  
            nn.Conv2d(curr_dim, output_channels, kernel_size=7, padding=0),  
            nn.Sigmoid()  
        ])  
        
        self.model = nn.Sequential(*layers)  
        
    def forward(self, x):  
        return self.model(x)

class ResnetGenerator2(nn.Module):  
    def __init__(self, input_shape=(9, 128, 128),  
                 output_channels=1,  
                 dim=64,  
                 n_downsamplings=3,  
                 n_blocks=6,
                 norm_layer=nn.InstanceNorm2d):  
        super().__init__()  
        
        self.H, self.W = input_shape[1], input_shape[2]  
        
        # First encoder
        self.encoder1 = Encoder(input_shape[0], dim, n_downsamplings, max_dim=256, norm_layer=norm_layer)  
        encoded_dim = self.encoder1.out_channels  
        
        # Position Embedding  
        h_encoded = self.H // (2**n_downsamplings)  
        w_encoded = self.W // (2**n_downsamplings)  
        self.pos_embedding = nn.Parameter(torch.randn(1, encoded_dim, h_encoded, w_encoded))  
        
        # Cross-Modal Attention  
        self.cross_attn = CrossModalAttention(encoded_dim, encoded_dim)  
        
        # Second encoder
        self.encoder2 = Encoder(encoded_dim, encoded_dim, 1, max_dim=512, norm_layer=norm_layer)
        final_encoded_dim = self.encoder2.out_channels  
        
        # Residual Blocks
        self.res_blocks = nn.Sequential(  
            *[ResBlock(final_encoded_dim, norm_layer) for _ in range(n_blocks)]  
        )  
        
        # Decoder
        self.decoder = Decoder(final_encoded_dim, output_channels, n_downsamplings + 1, norm_layer)  
        
    def forward(self, x):  
        # First encoding  
        feat1 = self.encoder1(x)  
        
        # Cross attention with position embedding  
        B = x.shape[0]  
        pos_emb = self.pos_embedding.expand(B, -1, -1, -1)  
        attended_feat = self.cross_attn(feat1, pos_emb)  
        
        # Second encoding  
        feat2 = self.encoder2(attended_feat)  
        
        # Residual blocks  
        res_out = self.res_blocks(feat2)  
        
        # Decoding to get single channel output  
        output_1ch = self.decoder(res_out)  # B, 1, 128, 128  
        # output_1ch = torch.relu(output_1ch)

        # Repeat the channel 3 times  
        output_3ch = output_1ch.repeat(1, 3, 1, 1)  # B, 3, 128, 128  
        
        return output_3ch

class VGGLoss(nn.Module):  
    def __init__(self):  
        super(VGGLoss, self).__init__()  
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(  
            *list(vgg.features.children())[:19]  
        ).eval()  
        for param in self.features.parameters():  
            param.requires_grad = False  
            
    def forward(self, x, y):
        # Handle single channel input
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.size(1) == 1:
            y = y.repeat(1, 3, 1, 1)
            
        x_vgg = self.features(x)  
        y_vgg = self.features(y)  
        return F.mse_loss(x_vgg, y_vgg)     