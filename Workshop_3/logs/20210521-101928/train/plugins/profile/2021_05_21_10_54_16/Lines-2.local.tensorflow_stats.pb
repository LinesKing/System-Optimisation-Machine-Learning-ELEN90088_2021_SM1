"?e
BHostIDLE"IDLE1    ?p?@A    ?p?@awg?J????iwg?J?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a|b?????i??(?2???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      ?@9      ?@A      ?@I      ?@aD??"!j?i?o?@M???Unknown
iHostWriteSummary"WriteSummary(1      ?@9      ?@A      ?@I      ?@aD??"!j?i??c3g???Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      <@9      <@A      <@I      <@a???̙g?i???0?~???Unknown
uHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      9@9      9@A      9@I      9@a?D?ve?i???ߓ???Unknown
vHost_FusedMatMul"sequential_38/dense_118/Relu(1      7@9      7@A      7@I      7@al|D?bc?iԻÏB????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      7@9      7@A      6@I      6@aQ?? ?b?i?S??͹???Unknown
?	HostMatMul",gradient_tape/sequential_38/dense_119/MatMul(1      0@9      0@A      0@I      0@a_???Z?iؓ?%J????Unknown
V
HostSum"Sum_2(1      ,@9      ,@A      ,@I      ,@a???̙W?i?2????Unknown
?HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      ,@9      ,@A      ,@I      ,@a???̙W?iڃ???????Unknown
vHost_FusedMatMul"sequential_38/dense_119/Relu(1      ,@9      ,@A      ,@I      ,@a???̙W?i??
ٰ????Unknown
?HostReadVariableOp".sequential_38/dense_119/BiasAdd/ReadVariableOp(1      *@9      *@A      *@I      *@a?(7>?U?i܏&??????Unknown
XHostCast"Cast_4(1      &@9      &@A      &@I      &@aQ?? ?R?i?[???????Unknown
?HostMatMul".gradient_tape/sequential_38/dense_119/MatMul_1(1      &@9      &@A      &@I      &@aQ?? ?R?i?'1???Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a?Q??P?i?C?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a?Q??P?i??k????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?`XN?i??C?? ???Unknown
dHostDataset"Iterator::Model(1     ?C@9     ?C@A       @I       @a_???J?i???`'???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a_???J?i?;R".???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a_???J?i?[?\?4???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a???̙G?i????:???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a???̙G?i??EC?@???Unknown?
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a???̙G?i?|??F???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a???̙G?i?K?)wL???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_38/dense_120/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???̙G?i????]R???Unknown
?HostMatMul",gradient_tape/sequential_38/dense_120/MatMul(1      @9      @A      @I      @a???̙G?i??DX???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?`??:D?i??R]???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?`??:D?i?s?gab???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?`??:D?i???pg???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?Q??@?i??c??k???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?Q??@?i?????o???Unknown
v!HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?Q??@?i?'??t???Unknown
?"Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?Q??@?i?? ?Kx???Unknown
?#HostBiasAddGrad"9gradient_tape/sequential_38/dense_118/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?Q??@?i????|???Unknown
?$HostBiasAddGrad"9gradient_tape/sequential_38/dense_119/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?Q??@?i??Io?????Unknown
y%Host_FusedMatMul"sequential_38/dense_120/BiasAdd(1      @9      @A      @I      @a?Q??@?i???S?????Unknown
t&HostSigmoid"sequential_38/dense_120/Sigmoid(1      @9      @A      @I      @a?Q??@?i?kr8'????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a_???:?i???U?????Unknown
V(HostMean"Mean(1      @9      @A      @I      @a_???:?i???r?????Unknown
v)HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a_???:?i?=?D????Unknown
~*HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a_???:?i?????????Unknown
~+HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a_???:?i?;??????Unknown
?,HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a_???:?i???a????Unknown
?-HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a_???:?i?[K?????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a?`??:4?i?>[H????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?`??:4?i??0?ϥ???Unknown
\0HostGreater"Greater(1      @9      @A      @I      @a?`??:4?i?_#W????Unknown
s1HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?`??:4?i?]ު???Unknown
?2HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?`??:4?i???e????Unknown
j3HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?`??:4?i?c??????Unknown
r4HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a?`??:4?i??^t????Unknown
v5HostExp"%binary_crossentropy/logistic_loss/Exp(1      @9      @A      @I      @a?`??:4?i?????????Unknown
?6HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?`??:4?i?g?
?????Unknown
z7HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?`??:4?i??`
????Unknown
v8HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?`??:4?i?????????Unknown
v9HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?`??:4?i?k?????Unknown
?:HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a?`??:4?i??b?????Unknown
b;HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?`??:4?i?Ð?'????Unknown
?<HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?`??:4?i?o??????Unknown
?=HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a?`??:4?i?vd6????Unknown
?>HostMatMul",gradient_tape/sequential_38/dense_118/MatMul(1      @9      @A      @I      @a?`??:4?i??h??????Unknown
??HostMatMul".gradient_tape/sequential_38/dense_120/MatMul_1(1      @9      @A      @I      @a?`??:4?i?s[E????Unknown
?@HostReadVariableOp".sequential_38/dense_118/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?`??:4?i?Nf?????Unknown
?AHostReadVariableOp"-sequential_38/dense_119/MatMul/ReadVariableOp(1      @9      @A      @I      @a?`??:4?i??@?S????Unknown
?BHostReadVariableOp"-sequential_38/dense_120/MatMul/ReadVariableOp(1      @9      @A      @I      @a?`??:4?i?w3?????Unknown
tCHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a_???*?i??ՠ?????Unknown
VDHostCast"Cast(1       @9       @A       @I       @a_???*?i?w/:????Unknown
XEHostCast"Cast_3(1       @9       @A       @I       @a_???*?i????????Unknown
XFHostEqual"Equal(1       @9       @A       @I       @a_???*?i???L?????Unknown
|GHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a_???*?i?_\?H????Unknown
dHHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a_???*?i?'?i?????Unknown
}IHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a_???*?i?????????Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a_???*?i??A?W????Unknown
wKHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a_???*?i??????Unknown
?LHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a_???*?i?G???????Unknown
xMHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a_???*?i?'3f????Unknown
?NHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a_???*?i????????Unknown
?OHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a_???*?i??jP?????Unknown
?PHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a_???*?i?g?t????Unknown
~QHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a_???*?i?/?m$????Unknown
?RHostReluGrad".gradient_tape/sequential_38/dense_118/ReluGrad(1       @9       @A       @I       @a_???*?i??O??????Unknown
?SHostReluGrad".gradient_tape/sequential_38/dense_119/ReluGrad(1       @9       @A       @I       @a_???*?i?????????Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a_????i??BR[????Unknown
XUHostCast"Cast_5(1      ??9      ??A      ??I      ??a_????i???3????Unknown
aVHostIdentity"Identity(1      ??9      ??A      ??I      ??a_????i?k??
????Unknown?
?WHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a_????i?O5??????Unknown
TXHostMul"Mul(1      ??9      ??A      ??I      ??a_????i?3?o?????Unknown
uYHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a_????i??6?????Unknown
wZHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a_????i??'?i????Unknown
y[HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a_????i??x?A????Unknown
?\Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a_????i??Ɍ????Unknown
?]HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a_????i??T?????Unknown
?^HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a_????i??k?????Unknown
?_HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a_????i?o???????Unknown
?`HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a_????i?S?x????Unknown
?aHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a_????i?7^qP????Unknown
?bHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      ??9      ??A      ??I      ??a_????i??8(????Unknown
?cHostReadVariableOp"-sequential_38/dense_118/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a_????i?????????Unknown
?dHostReadVariableOp".sequential_38/dense_120/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a_????i?q??k ???Unknown
WeHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?q??k ???Unknown
_fHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(i?q??k ???Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?q??k ???Unknown
YhHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?q??k ???Unknown*?d
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a@?!%????i@?!%?????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      ?@9      ?@A      ?@I      ?@au4ӻ\??i頻?M???Unknown
iHostWriteSummary"WriteSummary(1      ?@9      ?@A      ?@I      ?@au4ӻ\??i?DU?????Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      <@9      <@A      <@I      <@a/?-??iKu?P?????Unknown
uHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      9@9      9@A      9@I      9@aE?׺????i3dN?]???Unknown
vHost_FusedMatMul"sequential_38/dense_118/Relu(1      7@9      7@A      7@I      7@aT%A??i@??VQ????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      7@9      7@A      6@I      6@a\[??J??i???x???Unknown
?HostMatMul",gradient_tape/sequential_38/dense_119/MatMul(1      0@9      0@A      0@I      0@a??,?%??iT̐9????Unknown
V	HostSum"Sum_2(1      ,@9      ,@A      ,@I      ,@a/?-??itl?G>5???Unknown
?
HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      ,@9      ,@A      ,@I      ,@a/?-??iф?B????Unknown
vHost_FusedMatMul"sequential_38/dense_119/Relu(1      ,@9      ,@A      ,@I      ,@a/?-??i.? ?G????Unknown
?HostReadVariableOp".sequential_38/dense_119/BiasAdd/ReadVariableOp(1      *@9      *@A      *@I      *@a>?<?n??i;?q?7???Unknown
XHostCast"Cast_4(1      &@9      &@A      &@I      &@a\[??J??i?4,9+|???Unknown
?HostMatMul".gradient_tape/sequential_38/dense_119/MatMul_1(1      &@9      &@A      &@I      &@a\[??J??i??S????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@aՎ??eo?i3Y?K2 ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@aՎ??eo?iQ??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?f,??J|?i1?h?w???Unknown
dHostDataset"Iterator::Model(1     ?C@9     ?C@A       @I       @a??,?%y?i?cC??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a??,?%y?i??>????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a??,?%y?i????????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a/?-v?i?ԃG?:???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a/?-v?i????f???Unknown?
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a/?-v?i#????????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a/?-v?iQ?-Z?????Unknown
?HostBiasAddGrad"9gradient_tape/sequential_38/dense_120/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a/?-v?i???????Unknown
?HostMatMul",gradient_tape/sequential_38/dense_120/MatMul(1      @9      @A      @I      @a/?-v?i?J????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @aM?rap?r?i???P<???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aM?rap?r?ik???	b???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aM?rap?r?iJÒ????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aՎ??eoo?iق?2????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aՎ??eoo?ihB??????Unknown
v HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aՎ??eoo?i?z?????Unknown
?!Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @aՎ??eoo?i??qK????Unknown
?"HostBiasAddGrad"9gradient_tape/sequential_38/dense_118/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aՎ??eoo?i?i??$???Unknown
?#HostBiasAddGrad"9gradient_tape/sequential_38/dense_119/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aՎ??eoo?i?@a_D???Unknown
y$Host_FusedMatMul"sequential_38/dense_120/BiasAdd(1      @9      @A      @I      @aՎ??eoo?i3 Y}?c???Unknown
t%HostSigmoid"sequential_38/dense_120/Sigmoid(1      @9      @A      @I      @aՎ??eoo?i¿P?=????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??,?%i?iY}?c????Unknown
V'HostMean"Mean(1      @9      @A      @I      @a??,?%i?i@򩹉????Unknown
v(HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??,?%i?i?֤?????Unknown
~)HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a??,?%i?i?$??????Unknown
~*HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a??,?%i?i??/{? ???Unknown
?+HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??,?%i?i<W\f!???Unknown
?,HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a??,?%i?i{??QG3???Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @aM?rap?b?ijc??#F???Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aM?rap?b?iY?K2 Y???Unknown
\/HostGreater"Greater(1      @9      @A      @I      @aM?rap?b?iHI???k???Unknown
s0HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @aM?rap?b?i7??~???Unknown
?1HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aM?rap?b?i&/p??????Unknown
j2HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @aM?rap?b?i???q????Unknown
r3HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @aM?rap?b?i3dN????Unknown
v4HostExp"%binary_crossentropy/logistic_loss/Exp(1      @9      @A      @I      @aM?rap?b?i????*????Unknown
?5HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aM?rap?b?i???D????Unknown
z6HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @aM?rap?b?i?mW??????Unknown
v7HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @aM?rap?b?i???%????Unknown
v8HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aM?rap?b?i?S?????Unknown
?9HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @aM?rap?b?i??{y(???Unknown
b:HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aM?rap?b?i?9?vU;???Unknown
?;HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aM?rap?b?i|?>?1N???Unknown
?<HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @aM?rap?b?ik?Wa???Unknown
?=HostMatMul",gradient_tape/sequential_38/dense_118/MatMul(1      @9      @A      @I      @aM?rap?b?iZ???s???Unknown
?>HostMatMul".gradient_tape/sequential_38/dense_120/MatMul_1(1      @9      @A      @I      @aM?rap?b?iIc8ǆ???Unknown
??HostReadVariableOp".sequential_38/dense_118/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aM?rap?b?i8xĨ?????Unknown
?@HostReadVariableOp"-sequential_38/dense_119/MatMul/ReadVariableOp(1      @9      @A      @I      @aM?rap?b?i'?%?????Unknown
?AHostReadVariableOp"-sequential_38/dense_120/MatMul/ReadVariableOp(1      @9      @A      @I      @aM?rap?b?i^??\????Unknown
tBHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??,?%Y?i???????Unknown
VCHostCast"Cast(1       @9       @A       @I       @a??,?%Y?iV??t?????Unknown
XDHostCast"Cast_3(1       @9       @A       @I       @a??,?%Y?i?CJj????Unknown
XEHostEqual"Equal(1       @9       @A       @I       @a??,?%Y?i???_?????Unknown
|FHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??,?%Y?i6?vU;????Unknown
dGHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??,?%Y?i?)K?
???Unknown
}HHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a??,?%Y?ivv?@a???Unknown
`IHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??,?%Y?i?96?#???Unknown
wJHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??,?%Y?i??+?0???Unknown
?KHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??,?%Y?iV\f!=???Unknown
xLHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a??,?%Y?i????I???Unknown
?MHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a??,?%Y?i???@V???Unknown
?NHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??,?%Y?i6B)?b???Unknown
?OHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a??,?%Y?i֎??eo???Unknown
~PHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a??,?%Y?iv?U??{???Unknown
?QHostReluGrad".gradient_tape/sequential_38/dense_118/ReluGrad(1       @9       @A       @I       @a??,?%Y?i(?⋈???Unknown
?RHostReluGrad".gradient_tape/sequential_38/dense_119/ReluGrad(1       @9       @A       @I       @a??,?%Y?i?t??????Unknown
vSHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??,?%I?i?MSh????Unknown
XTHostCast"Cast_5(1      ??9      ??A      ??I      ??a??,?%I?iV?α????Unknown
aUHostIdentity"Identity(1      ??9      ??A      ??I      ??a??,?%I?i???H?????Unknown?
?VHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a??,?%I?i???D????Unknown
TWHostMul"Mul(1      ??9      ??A      ??I      ??a??,?%I?iF4z>?????Unknown
uXHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??,?%I?i?ZE?׺???Unknown
wYHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??,?%I?i??4!????Unknown
yZHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??,?%I?i6?ۮj????Unknown
?[Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??,?%I?i?ͦ)?????Unknown
?\HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a??,?%I?i??q??????Unknown
?]HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??,?%I?i&=G????Unknown
?^HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??,?%I?iv@??????Unknown
?_HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??,?%I?i?f??????Unknown
?`HostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a??,?%I?i???#????Unknown
?aHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      ??9      ??A      ??I      ??a??,?%I?if?i
m????Unknown
?bHostReadVariableOp"-sequential_38/dense_118/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??,?%I?i??4??????Unknown
?cHostReadVariableOp".sequential_38/dense_120/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??,?%I?i     ???Unknown
WdHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i     ???Unknown
_eHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(i     ???Unknown
[fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown
YgHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown2CPU