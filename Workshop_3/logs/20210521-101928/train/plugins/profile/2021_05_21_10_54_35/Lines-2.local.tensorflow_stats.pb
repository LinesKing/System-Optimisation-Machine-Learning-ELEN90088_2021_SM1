"?f
BHostIDLE"IDLE1     ??@A     ??@a?/0t????i?/0t?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a+|_?????i?o{e???Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      A@9      A@A      A@I      A@a?"??~^l?i7?_?ف???Unknown
iHostWriteSummary"WriteSummary(1      =@9      =@A      =@I      =@a?R??z2h?i?YWi????Unknown?
vHost_FusedMatMul"sequential_40/dense_124/Relu(1      :@9      :@A      :@I      :@aJ?8
??e?i,?a?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      9@9      9@A      9@I      9@a3??d?i>?'?????Unknown
^HostGatherV2"GatherV2(1      0@9      0@A      0@I      0@a?2?I?Z?i?:??????Unknown
yHost_FusedMatMul"sequential_40/dense_126/BiasAdd(1      .@9      .@A      .@I      .@a???FY?i????w????Unknown
v	HostMul"%binary_crossentropy/logistic_loss/mul(1      ,@9      ,@A      ,@I      ,@aw??\W?i??G&????Unknown
?
HostMatMul".gradient_tape/sequential_40/dense_125/MatMul_1(1      *@9      *@A      *@I      *@aJ?8
??U?i??????Unknown
\HostGreater"Greater(1      (@9      (@A      (@I      (@a??kwT?i???X????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      &@9      &@A      &@I      &@a?a??B[R?i??(?/???Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      &@9      &@A      &@I      &@a?a??B[R?i???]???Unknown
?HostMatMul",gradient_tape/sequential_40/dense_126/MatMul(1      $@9      $@A      $@I      $@a?A?/?P?i?0??????Unknown
vHost_FusedMatMul"sequential_40/dense_125/Relu(1      "@9      "@A      "@I      "@a+C?!?	N?i??o8!???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?2?I?J?i8????'???Unknown?
?HostMatMul",gradient_tape/sequential_40/dense_124/MatMul(1       @9       @A       @I       @a?2?I?J?i??b??.???Unknown
dHostDataset"Iterator::Model(1      @@9      @@A      @I      @aw??\G?i?b??h4???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aw??\G?i???$@:???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aw??\G?i???\@???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @aw??\G?i}?
??E???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aw??\G?in?4??K???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a??kwD?i??k?P???Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a??kwD?i0???U???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a??kwD?i??Ŧ?Z???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a??kwD?i?ӠD?_???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a??kwD?iS?{??d???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_40/dense_125/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??kwD?i??V??i???Unknown
?HostMatMul",gradient_tape/sequential_40/dense_125/MatMul(1      @9      @A      @I      @a??kwD?i?1?n???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?A?/?@?i叽!?r???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?A?/?@?i?_I%)w???Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?A?/?@?i?/?(U{???Unknown
?!HostBiasAddGrad"9gradient_tape/sequential_40/dense_124/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?A?/?@?iU?`,????Unknown
?"HostMatMul".gradient_tape/sequential_40/dense_126/MatMul_1(1      @9      @A      @I      @a?A?/?@?i%??/?????Unknown
?#HostReadVariableOp"-sequential_40/dense_125/MatMul/ReadVariableOp(1      @9      @A      @I      @a?A?/?@?i??x3ه???Unknown
?$HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?2?I?:?i5E??/????Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?2?I?:?iu???????Unknown
v&HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?2?I?:?i??.oܑ???Unknown
?'HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?2?I?:?i?7k?2????Unknown
?(HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?2?I?:?i5ާA?????Unknown
?)Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?2?I?:?iu???ߛ???Unknown
?*HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?2?I?:?i?*!6????Unknown
?+HostBiasAddGrad"9gradient_tape/sequential_40/dense_126/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?2?I?:?i??]}?????Unknown
t,HostSigmoid"sequential_40/dense_126/Sigmoid(1      @9      @A      @I      @a?2?I?:?i5w???????Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a??kw4?i????c????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??kw4?i?pu??????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??kw4?iE?bSe????Unknown
V0HostMean"Mean(1      @9      @A      @I      @a??kw4?i?iP"?????Unknown
s1HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a??kw4?i??=?f????Unknown
z2HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??kw4?iUc+??????Unknown
v3HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??kw4?i??h????Unknown
~4HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a??kw4?i?\^?????Unknown
b5HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??kw4?ie??,j????Unknown
?6HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a??kw4?iV???????Unknown
?7HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a??kw4?i????k????Unknown
?8HostReadVariableOp"-sequential_40/dense_124/MatMul/ReadVariableOp(1      @9      @A      @I      @a??kw4?iuO???????Unknown
?9HostReadVariableOp".sequential_40/dense_126/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??kw4?i%̩hm????Unknown
?:HostReadVariableOp"-sequential_40/dense_126/MatMul/ReadVariableOp(1      @9      @A      @I      @a??kw4?i?H?7?????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?2?I?*?i??5l?????Unknown
V<HostCast"Cast(1       @9       @A       @I       @a?2?I?*?i?ӠD????Unknown
X=HostCast"Cast_3(1       @9       @A       @I       @a?2?I?*?i5Br??????Unknown
X>HostCast"Cast_5(1       @9       @A       @I       @a?2?I?*?iU?
?????Unknown
X?HostEqual"Equal(1       @9       @A       @I       @a?2?I?*?iu??>F????Unknown
|@HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?2?I?*?i?;Ms?????Unknown
dAHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?2?I?*?i??맜????Unknown
jBHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?2?I?*?i????G????Unknown
rCHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?2?I?*?i?4(?????Unknown
?DHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a?2?I?*?i??E?????Unknown
vEHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?2?I?*?i5?dzI????Unknown
?FHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?2?I?*?iU.??????Unknown
`GHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?2?I?*?iu????????Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?2?I?*?i???K????Unknown
xIHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?2?I?*?i?'?L?????Unknown
~JHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?2?I?*?i?z|??????Unknown
?KHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?2?I?*?i???L????Unknown
?LHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?2?I?*?i!???????Unknown
?MHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?2?I?*?i5tW?????Unknown
?NHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?2?I?*?iU??SN????Unknown
?OHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a?2?I?*?iu???????Unknown
~PHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?2?I?*?i?m2??????Unknown
?QHostReluGrad".gradient_tape/sequential_40/dense_125/ReluGrad(1       @9       @A       @I       @a?2?I?*?i????O????Unknown
?RHostReadVariableOp".sequential_40/dense_125/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?2?I?*?i?o&?????Unknown
vSHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?2?I??ie=???????Unknown
XTHostCast"Cast_4(1      ??9      ??A      ??I      ??a?2?I??i?f[?????Unknown
aUHostIdentity"Identity(1      ??9      ??A      ??I      ??a?2?I??i??\?{????Unknown?
?VHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??a?2?I??i???Q????Unknown
?WHostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor(1      ??9      ??A      ??I      ??a?2?I??i???)'????Unknown
TXHostMul"Mul(1      ??9      ??A      ??I      ??a?2?I??i5J??????Unknown
uYHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?2?I??i?6?^?????Unknown
vZHostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a?2?I??iU`???????Unknown
}[HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?2?I??i??7?}????Unknown
u\HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?2?I??iu??-S????Unknown
w]HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?2?I??i???(????Unknown
y^HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?2?I??i?%b?????Unknown
?_HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a?2?I??i%0t??????Unknown
?`HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?2?I??i?YÖ?????Unknown
?aHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?2?I??iE?1????Unknown
?bHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?2?I??iլa?T????Unknown
?cHostReluGrad".gradient_tape/sequential_40/dense_124/ReluGrad(1      ??9      ??A      ??I      ??a?2?I??ieְe*????Unknown
?dHostReadVariableOp".sequential_40/dense_124/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?2?I??i?????????Unknown
ieHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
WfHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
WgHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i?????????Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
[iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown*?e
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a33333???i33333????Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      A@9      A@A      A@I      A@a333333??i????̌???Unknown
iHostWriteSummary"WriteSummary(1      =@9      =@A      =@I      =@a333333??igffffF???Unknown?
vHost_FusedMatMul"sequential_40/dense_124/Relu(1      :@9      :@A      :@I      :@a?????̔?i?????????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      9@9      9@A      9@I      9@a      ??i????̌???Unknown
^HostGatherV2"GatherV2(1      0@9      0@A      0@I      0@a????????i33333????Unknown
yHost_FusedMatMul"sequential_40/dense_126/BiasAdd(1      .@9      .@A      .@I      .@a      ??i33333S???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      ,@9      ,@A      ,@I      ,@affffff??i????̬???Unknown
?	HostMatMul".gradient_tape/sequential_40/dense_125/MatMul_1(1      *@9      *@A      *@I      *@a?????̄?i      ???Unknown
\
HostGreater"Greater(1      (@9      (@A      (@I      (@a333333??i?????L???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      &@9      &@A      &@I      &@a????????i33333????Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      &@9      &@A      &@I      &@a????????i?????????Unknown
?HostMatMul",gradient_tape/sequential_40/dense_126/MatMul(1      $@9      $@A      $@I      $@a      ??i????????Unknown
vHost_FusedMatMul"sequential_40/dense_125/Relu(1      "@9      "@A      "@I      "@a??????|?i33333S???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a??????y?ifffff????Unknown?
?HostMatMul",gradient_tape/sequential_40/dense_124/MatMul(1       @9       @A       @I       @a??????y?i?????????Unknown
dHostDataset"Iterator::Model(1      @@9      @@A      @I      @affffffv?ifffff????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @affffffv?i33333???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @affffffv?i     @???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @affffffv?i?????l???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @affffffv?i?????????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a333333s?i     ????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a333333s?ifffff????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a333333s?i????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a333333s?i233333???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a333333s?i?????Y???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_40/dense_125/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a333333s?i????????Unknown
?HostMatMul",gradient_tape/sequential_40/dense_125/MatMul(1      @9      @A      @I      @a333333s?idffff????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a      p?idffff????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a      p?idffff????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a      p?idffff???Unknown
? HostBiasAddGrad"9gradient_tape/sequential_40/dense_124/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a      p?idffff&???Unknown
?!HostMatMul".gradient_tape/sequential_40/dense_126/MatMul_1(1      @9      @A      @I      @a      p?idffffF???Unknown
?"HostReadVariableOp"-sequential_40/dense_125/MatMul/ReadVariableOp(1      @9      @A      @I      @a      p?idfffff???Unknown
?#HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??????i?i????????Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a??????i?i?????????Unknown
v%HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??????i?i23333????Unknown
?&HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??????i?i?????????Unknown
?'HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??????i?ifffff????Unknown
?(Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??????i?i      ???Unknown
?)HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??????i?i????????Unknown
?*HostBiasAddGrad"9gradient_tape/sequential_40/dense_126/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??????i?i433333???Unknown
t+HostSigmoid"sequential_40/dense_126/Sigmoid(1      @9      @A      @I      @a??????i?i?????L???Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a333333c?i    `???Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a333333c?i43333s???Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a333333c?igffff????Unknown
V/HostMean"Mean(1      @9      @A      @I      @a333333c?i?????????Unknown
s0HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a333333c?i????̬???Unknown
z1HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a333333c?i     ????Unknown
v2HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a333333c?i33333????Unknown
~3HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a333333c?ifffff????Unknown
b4HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a333333c?i?????????Unknown
?5HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a333333c?i????????Unknown
?6HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a333333c?i????????Unknown
?7HostReadVariableOp"-sequential_40/dense_124/MatMul/ReadVariableOp(1      @9      @A      @I      @a333333c?i233333???Unknown
?8HostReadVariableOp".sequential_40/dense_126/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a333333c?ieffffF???Unknown
?9HostReadVariableOp"-sequential_40/dense_126/MatMul/ReadVariableOp(1      @9      @A      @I      @a333333c?i?????Y???Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a??????Y?iefffff???Unknown
V;HostCast"Cast(1       @9       @A       @I       @a??????Y?i23333s???Unknown
X<HostCast"Cast_3(1       @9       @A       @I       @a??????Y?i????????Unknown
X=HostCast"Cast_5(1       @9       @A       @I       @a??????Y?i????̌???Unknown
X>HostEqual"Equal(1       @9       @A       @I       @a??????Y?i?????????Unknown
|?HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??????Y?ifffff????Unknown
d@HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??????Y?i33333????Unknown
jAHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a??????Y?i     ????Unknown
rBHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a??????Y?i?????????Unknown
?CHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a??????Y?i?????????Unknown
vDHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a??????Y?igffff????Unknown
?EHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a??????Y?i43333????Unknown
`FHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??????Y?i     ???Unknown
wGHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??????Y?i????????Unknown
xHHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a??????Y?i????????Unknown
~IHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a??????Y?ihffff&???Unknown
?JHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a??????Y?i533333???Unknown
?KHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a??????Y?i    @???Unknown
?LHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??????Y?i?????L???Unknown
?MHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a??????Y?i?????Y???Unknown
?NHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a??????Y?iifffff???Unknown
~OHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a??????Y?i63333s???Unknown
?PHostReluGrad".gradient_tape/sequential_40/dense_125/ReluGrad(1       @9       @A       @I       @a??????Y?i    ????Unknown
?QHostReadVariableOp".sequential_40/dense_125/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??????Y?i????̌???Unknown
vRHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a??????I?i63333????Unknown
XSHostCast"Cast_4(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
aTHostIdentity"Identity(1      ??9      ??A      ??I      ??a??????I?i    ????Unknown?
?UHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??a??????I?ihffff????Unknown
?VHostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor(1      ??9      ??A      ??I      ??a??????I?i????̬???Unknown
TWHostMul"Mul(1      ??9      ??A      ??I      ??a??????I?i43333????Unknown
uXHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
vYHostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a??????I?i     ????Unknown
}ZHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a??????I?ifffff????Unknown
u[HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
w\HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??????I?i23333????Unknown
y]HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
?^HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
?_HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a??????I?idffff????Unknown
?`HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
?aHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a??????I?i03333????Unknown
?bHostReluGrad".gradient_tape/sequential_40/dense_124/ReluGrad(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
?cHostReadVariableOp".sequential_40/dense_124/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
idHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
WeHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
WfHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i?????????Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown2CPU