"?p
BHostIDLE"IDLE1     /?@A     /?@a???%???i???%????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     8?@9     8?@A     8?@I     8?@a??	6α?i?(?U????Unknown?
iHostWriteSummary"WriteSummary(1      @@9      @@A      @@I      @@a??A։h?i??,p5???Unknown?
dHostDataset"Iterator::Model(1      =@9      =@A      =@I      =@a7?_+?<f?i?C?K???Unknown
vHost_FusedMatMul"sequential_96/dense_297/Relu(1      =@9      =@A      =@I      =@a7?_+?<f?iDln ?a???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      =@9      =@A      9@I      9@a?"/c?+c?ig?ѯu???Unknown
?HostMatMul",gradient_tape/sequential_96/dense_298/MatMul(1      5@9      5@A      5@I      5@aT???t`?i	?l$0????Unknown
?HostMatMul".gradient_tape/sequential_96/dense_299/MatMul_1(1      5@9      5@A      5@I      5@aT???t`?i???J????Unknown
?	HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      :@9      :@A      4@I      4@ao??K?^?i-?𾠤???Unknown
~
HostRealDiv")gradient_tape/binary_crossentropy/truediv(1      2@9      2@A      2@I      2@a???	?[?ioeuGn????Unknown
?HostReadVariableOp"-sequential_96/dense_298/MatMul/ReadVariableOp(1      2@9      2@A      2@I      2@a???	?[?i????;????Unknown
gHostStridedSlice"strided_slice(1      0@9      0@A      0@I      0@a??A։X?i???????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      .@9      .@A      .@I      .@aS?k?8W?i???W????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      ,@9      ,@A      ,@I      ,@a?Sy?xU?iWaF??????Unknown
VHostSum"Sum_2(1      ,@9      ,@A      ,@I      ,@a?Sy?xU?i?y????Unknown
vHost_FusedMatMul"sequential_96/dense_299/Relu(1      (@9      (@A      (@I      (@a?#?`gR?i??[??????Unknown
XHostCast"Cast_5(1      $@9      $@A      $@I      $@ao??K?N?i?P?X????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@ao??K?N?i?D????Unknown
?HostMatMul".gradient_tape/sequential_96/dense_298/MatMul_1(1      $@9      $@A      $@I      $@ao??K?N?i]9ܮ???Unknown
^HostGatherV2"GatherV2(1      "@9      "@A      "@I      "@a???	?K?i~u{?????Unknown
?HostMatMul",gradient_tape/sequential_96/dense_300/MatMul(1      "@9      "@A      "@I      "@a???	?K?i???d|???Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a??A։H?i?CNڞ!???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1       @9       @A       @I       @a??A։H?i???O?'???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1       @9       @A       @I       @a??A։H?i?o??-???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a??A։H?i?f?:4???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a??A։H?i?Ǐ?(:???Unknown
?HostMatMul",gradient_tape/sequential_96/dense_299/MatMul(1       @9       @A       @I       @a??A։H?i?( &K@???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?Sy?xE?i?}?L?E???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?Sy?xE?ig??sK???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a?Sy?xE?iH'??eP???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?#?`gB?i	p?r?T???Unknown
| HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?#?`gB?iʸK?Y???Unknown
?!HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?#?`gB?i?@#3^???Unknown
?"HostBiasAddGrad"9gradient_tape/sequential_96/dense_299/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?#?`gB?iLJl??b???Unknown
x#HostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?O@9     ?O@A      @I      @ao??K?>?i??愢f???Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @ao??K?>?i??`xj???Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @ao??K?>?i/ ۗMn???Unknown
?&HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @ao??K?>?i?<U!#r???Unknown
b'HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @ao??K?>?iqyϪ?u???Unknown
?(HostBiasAddGrad"9gradient_tape/sequential_96/dense_300/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ao??K?>?i?I4?y???Unknown
\)HostGreater"Greater(1      @9      @A      @I      @a??A։8?i??o?|???Unknown
?*HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a??A։8?iک????Unknown
v+HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??A։8?i?G??????Unknown
~,HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a??A։8?ixj????Unknown
v-HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??A։8?i??2Z$????Unknown
?.HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??A։8?i???5????Unknown
?/HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??A։8?i?	??F????Unknown
?0HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??A։8?i:?
X????Unknown
?1HostMatMul",gradient_tape/sequential_96/dense_297/MatMul(1      @9      @A      @I      @a??A։8?i?jSEi????Unknown
?2HostBiasAddGrad"9gradient_tape/sequential_96/dense_298/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??A։8?i??z????Unknown
?3HostMatMul".gradient_tape/sequential_96/dense_300/MatMul_1(1      @9      @A      @I      @a??A։8?i??㺋????Unknown
?4HostReadVariableOp"-sequential_96/dense_297/MatMul/ReadVariableOp(1      @9      @A      @I      @a??A։8?i????????Unknown
?5HostReadVariableOp".sequential_96/dense_298/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??A։8?i?,t0?????Unknown
v6Host_FusedMatMul"sequential_96/dense_298/Relu(1      @9      @A      @I      @a??A։8?i]<k?????Unknown
y7Host_FusedMatMul"sequential_96/dense_300/BiasAdd(1      @9      @A      @I      @a??A։8?i???Ч???Unknown
t8HostSigmoid"sequential_96/dense_300/Sigmoid(1      @9      @A      @I      @a??A։8?i????????Unknown
t9HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?#?`g2?ir???.????Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a?#?`g2?i???{????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?#?`g2?i2+?ȱ???Unknown
?<HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      =@9      =@A      @I      @a?#?`g2?i?O%?????Unknown
V=HostMean"Mean(1      @9      @A      @I      @a?#?`g2?i?s;}b????Unknown
s>HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?#?`g2?iR?Qi?????Unknown
u?HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?#?`g2?i??gU?????Unknown
r@HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a?#?`g2?i?}AI????Unknown
?AHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?#?`g2?ir?-?????Unknown
zBHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?#?`g2?i?)??????Unknown
vCHostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?#?`g2?i2N?0????Unknown
vDHostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?#?`g2?i?r??|????Unknown
~EHostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?#?`g2?i?????????Unknown
?FHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?#?`g2?iR??????Unknown
?GHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a?#?`g2?i???c????Unknown
?HHostBiasAddGrad"9gradient_tape/sequential_96/dense_297/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?#?`g2?i/??????Unknown
?IHostReadVariableOp".sequential_96/dense_299/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?#?`g2?ir(E??????Unknown
?JHostReadVariableOp"-sequential_96/dense_299/MatMul/ReadVariableOp(1      @9      @A      @I      @a?#?`g2?i?L[zJ????Unknown
VKHostCast"Cast(1       @9       @A       @I       @a??A։(?ie??????Unknown
XLHostCast"Cast_3(1       @9       @A       @I       @a??A։(?iR}#?[????Unknown
XMHostEqual"Equal(1       @9       @A       @I       @a??A։(?i???R?????Unknown
|NHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??A։(?iҭ??l????Unknown
dOHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??A։(?i?O??????Unknown
jPHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a??A։(?iR޳*~????Unknown
?QHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a??A։(?i???????Unknown
`RHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??A։(?i?|e?????Unknown
wSHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??A։(?i'?????Unknown
?THostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??A։(?iR?D??????Unknown
xUHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a??A։(?i?W?=)????Unknown
?VHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??A։(?i?o۱????Unknown
?WHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a??A։(?i?px:????Unknown
?XHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a??A։(?iR???????Unknown
?YHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a??A։(?i??8?K????Unknown
?ZHostReluGrad".gradient_tape/sequential_96/dense_297/ReluGrad(1       @9       @A       @I       @a??A։(?i?МP?????Unknown
?[HostReluGrad".gradient_tape/sequential_96/dense_299/ReluGrad(1       @9       @A       @I       @a??A։(?i? ?\????Unknown
?\HostReadVariableOp".sequential_96/dense_297/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??A։(?iRe??????Unknown
?]HostReadVariableOp".sequential_96/dense_300/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??A։(?i??(n????Unknown
?^HostReadVariableOp"-sequential_96/dense_300/MatMul/ReadVariableOp(1       @9       @A       @I       @a??A։(?i?1-??????Unknown
v_HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??A։?i?=??????Unknown
X`HostCast"Cast_4(1      ??9      ??A      ??I      ??a??A։?iJ?c????Unknown
aaHostIdentity"Identity(1      ??9      ??A      ??I      ??a??A։?i2VC?C????Unknown?
TbHostMul"Mul(1      ??9      ??A      ??I      ??a??A։?iRb? ????Unknown
vcHostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a??A։?irn?O?????Unknown
}dHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a??A։?i?zY??????Unknown
ueHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??A։?i???T????Unknown
wfHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??A։?iҒ?;????Unknown
ygHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??A։?i??o??????Unknown
?hHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a??A։?i?!١????Unknown
?iHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??A։?i2??'f????Unknown
?jHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a??A։?iRÅv*????Unknown
?kHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??A։?ir?7??????Unknown
?lHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??A։?i????????Unknown
?mHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??A։?i???bw????Unknown
?nHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a??A։?i??M?;????Unknown
?oHostReluGrad".gradient_tape/sequential_96/dense_298/ReluGrad(1      ??9      ??A      ??I      ??a??A։?i?????????Unknown
WpHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
[qHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown
]rHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(i?????????Unknown*?p
uHostFlushSummaryWriter"FlushSummaryWriter(1     8?@9     8?@A     8?@I     8?@a%??hQ??i%??hQ???Unknown?
iHostWriteSummary"WriteSummary(1      @@9      @@A      @@I      @@a?hQ?ݗ?ij~"????Unknown?
dHostDataset"Iterator::Model(1      =@9      =@A      =@I      =@a?ր?蠕?i ??8????Unknown
vHost_FusedMatMul"sequential_96/dense_297/Relu(1      =@9      =@A      =@I      =@a?ր?蠕?i֋:~j???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      =@9      =@A      9@I      9@a?i]?2???i$wuA????Unknown
?HostMatMul",gradient_tape/sequential_96/dense_298/MatMul(1      5@9      5@A      5@I      5@aJ?s??R??i	G_??|???Unknown
?HostMatMul".gradient_tape/sequential_96/dense_299/MatMul_1(1      5@9      5@A      5@I      5@aJ?s??R??i?I??????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      :@9      :@A      4@I      4@a?Bb?Ս?i???N-q???Unknown
~	HostRealDiv")gradient_tape/binary_crossentropy/truediv(1      2@9      2@A      2@I      2@a??>{gي?iP????????Unknown
?
HostReadVariableOp"-sequential_96/dense_298/MatMul/ReadVariableOp(1      2@9      2@A      2@I      2@a??>{gي?i?????G???Unknown
gHostStridedSlice"strided_slice(1      0@9      0@A      0@I      0@a?hQ?݇?iI?Oo????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      .@9      .@A      .@I      .@a?	<?_??i+??? ???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      ,@9      ,@A      ,@I      ,@a???&????i?
??vT???Unknown
VHostSum"Sum_2(1      ,@9      ,@A      ,@I      ,@a???&????i??$??????Unknown
vHost_FusedMatMul"sequential_96/dense_299/Relu(1      (@9      (@A      (@I      (@as???D???i'=??????Unknown
XHostCast"Cast_5(1      $@9      $@A      $@I      $@a?Bb??}?i?c?A+???Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a?Bb??}?i2ƭ?f???Unknown
?HostMatMul".gradient_tape/sequential_96/dense_298/MatMul_1(1      $@9      $@A      $@I      $@a?Bb??}?i???G?????Unknown
^HostGatherV2"GatherV2(1      "@9      "@A      "@I      "@a??>{g?z?ic?I????Unknown
?HostMatMul",gradient_tape/sequential_96/dense_300/MatMul(1      "@9      "@A      "@I      "@a??>{g?z?i???????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a?hQ??w?i߼?H?=???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1       @9       @A       @I       @a?hQ??w?i??)?rm???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1       @9       @A       @I       @a?hQ??w?i?*?.????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a?hQ??w?iRanp?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?hQ??w?i#?Ӥ????Unknown
?HostMatMul",gradient_tape/sequential_96/dense_299/MatMul(1       @9       @A       @I       @a?hQ??w?i?β5`,???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a???&??t?i?? ,$V???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a???&??t?i??N"????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a???&??t?iٞ??????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @as???D?q?i?G??x????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @as???D?q?i??,E????Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @as???D?q?i0??????Unknown
?!HostBiasAddGrad"9gradient_tape/sequential_96/dense_299/BiasAdd/BiasAddGrad(1      @9      @A      @I      @as???D?q?iMC?@?8???Unknown
x"HostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?O@9     ?O@A      @I      @a?Bb??m?i??(^?V???Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?Bb??m?i??{?t???Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?Bb??m?ijs?]????Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?Bb??m?iY??2????Unknown
b&HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?Bb??m?i?.??????Unknown
?'HostBiasAddGrad"9gradient_tape/sequential_96/dense_300/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?Bb??m?iߐc??????Unknown
\(HostGreater"Greater(1      @9      @A      @I      @a?hQ??g?iH???????Unknown
?)HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?hQ??g?i??U????Unknown
v*HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?hQ??g?i?Vv3???Unknown
~+HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?hQ??g?i????SK???Unknown
v,HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?hQ??g?i??h1c???Unknown
?-HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?hQ??g?iU5J{???Unknown
?.HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?hQ??g?i?P???????Unknown
?/HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?hQ??g?i'l?|ʪ???Unknown
?0HostMatMul",gradient_tape/sequential_96/dense_297/MatMul(1      @9      @A      @I      @a?hQ??g?i??=.?????Unknown
?1HostBiasAddGrad"9gradient_tape/sequential_96/dense_298/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?hQ??g?i???߅????Unknown
?2HostMatMul".gradient_tape/sequential_96/dense_300/MatMul_1(1      @9      @A      @I      @a?hQ??g?ib?ߐc????Unknown
?3HostReadVariableOp"-sequential_96/dense_297/MatMul/ReadVariableOp(1      @9      @A      @I      @a?hQ??g?i??0BA
???Unknown
?4HostReadVariableOp".sequential_96/dense_298/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?hQ??g?i4???"???Unknown
v5Host_FusedMatMul"sequential_96/dense_298/Relu(1      @9      @A      @I      @a?hQ??g?i?Ӥ?9???Unknown
y6Host_FusedMatMul"sequential_96/dense_300/BiasAdd(1      @9      @A      @I      @a?hQ??g?i,$V?Q???Unknown
t7HostSigmoid"sequential_96/dense_300/Sigmoid(1      @9      @A      @I      @a?hQ??g?ioGu?i???Unknown
t8HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @as???D?a?i?rL?{???Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @as???D?a?i??n??????Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @as???D?a?i?k?j????Unknown
?;HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      =@9      =@A      @I      @as???D?a?i??hQ????Unknown
V<HostMean"Mean(1      @9      @A      @I      @as???D?a?i5ne`7????Unknown
s=HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @as???D?a?i?Bb?????Unknown
u>HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @as???D?a?iQ_?????Unknown
r?HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @as???D?a?i??[/?????Unknown
?@HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @as???D?a?im?Xt?
???Unknown
zAHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @as???D?a?i??U?????Unknown
vBHostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @as???D?a?i?iR??.???Unknown
vCHostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @as???D?a?i>OC?@???Unknown
~DHostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @as???D?a?i?L?iR???Unknown
?EHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @as???D?a?i3?H?Od???Unknown
?FHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @as???D?a?i??E6v???Unknown
?GHostBiasAddGrad"9gradient_tape/sequential_96/dense_297/BiasAdd/BiasAddGrad(1      @9      @A      @I      @as???D?a?iO?BW????Unknown
?HHostReadVariableOp".sequential_96/dense_299/BiasAdd/ReadVariableOp(1      @9      @A      @I      @as???D?a?i?d??????Unknown
?IHostReadVariableOp"-sequential_96/dense_299/MatMul/ReadVariableOp(1      @9      @A      @I      @as???D?a?ik9<??????Unknown
VJHostCast"Cast(1       @9       @A       @I       @a?hQ??W?i???׷???Unknown
XKHostCast"Cast_3(1       @9       @A       @I       @a?hQ??W?i?T???????Unknown
XLHostEqual"Equal(1       @9       @A       @I       @a?hQ??W?i??5k?????Unknown
|MHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?hQ??W?i;p?C?????Unknown
dNHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?hQ??W?i????????Unknown
jOHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?hQ??W?i??/??????Unknown
?PHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?hQ??W?iW??p????Unknown
`QHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?hQ??W?i???_???Unknown
wRHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?hQ??W?i?4)N???Unknown
?SHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?hQ??W?is??W=#???Unknown
xTHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?hQ??W?i'Pz0,/???Unknown
?UHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?hQ??W?i??"	;???Unknown
?VHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?hQ??W?i?k??	G???Unknown
?WHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?hQ??W?iC?s??R???Unknown
?XHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?hQ??W?i????^???Unknown
?YHostReluGrad".gradient_tape/sequential_96/dense_297/ReluGrad(1       @9       @A       @I       @a?hQ??W?i??k?j???Unknown
?ZHostReluGrad".gradient_tape/sequential_96/dense_299/ReluGrad(1       @9       @A       @I       @a?hQ??W?i_?mD?v???Unknown
?[HostReadVariableOp".sequential_96/dense_297/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?hQ??W?i0?????Unknown
?\HostReadVariableOp".sequential_96/dense_300/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?hQ??W?iǽ???????Unknown
?]HostReadVariableOp"-sequential_96/dense_300/MatMul/ReadVariableOp(1       @9       @A       @I       @a?hQ??W?i{KgΑ????Unknown
v^HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?hQ??G?iU??:?????Unknown
X_HostCast"Cast_4(1      ??9      ??A      ??I      ??a?hQ??G?i/???????Unknown
a`HostIdentity"Identity(1      ??9      ??A      ??I      ??a?hQ??G?i	 dx????Unknown?
TaHostMul"Mul(1      ??9      ??A      ??I      ??a?hQ??G?i?f?o????Unknown
vbHostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a?hQ??G?i???f????Unknown
}cHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?hQ??G?i??`X^????Unknown
udHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?hQ??G?iq;??U????Unknown
weHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?hQ??G?iK?	1M????Unknown
yfHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?hQ??G?i%?]?D????Unknown
?gHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?hQ??G?i??	<????Unknown
?hHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?hQ??G?i?Vv3????Unknown
?iHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?hQ??G?i??Z?*????Unknown
?jHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?hQ??G?i???N"????Unknown
?kHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?hQ??G?ig+?????Unknown
?lHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?hQ??G?iArW'????Unknown
?mHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?hQ??G?i???????Unknown
?nHostReluGrad".gradient_tape/sequential_96/dense_298/ReluGrad(1      ??9      ??A      ??I      ??a?hQ??G?i?????????Unknown
WoHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
[pHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown
]qHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(i?????????Unknown2CPU