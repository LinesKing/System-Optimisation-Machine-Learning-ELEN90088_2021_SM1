"?o
BHostIDLE"IDLE1    ?G?@A    ?G?@a?d?0b:??i?d?0b:???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a?8?u???i Ly?|???Unknown?
vHost_FusedMatMul"sequential_72/dense_223/Relu(1      A@9      A@A      A@I      A@as?*?h?i?X???????Unknown
dHostDataset"Iterator::Model(1      ;@9      ;@A      ;@I      ;@ac???&?c?i2??b????Unknown
iHostWriteSummary"WriteSummary(1      ;@9      ;@A      ;@I      ;@ac???&?c?i?՚?&????Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      9@9      9@A      8@I      8@a?$T???a?i *^¸????Unknown
?HostMatMul",gradient_tape/sequential_72/dense_224/MatMul(1      3@9      3@A      3@I      3@a?????[?i?!n??????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      1@9      1@A      1@I      1@as?*?X?i7(?????Unknown
^	HostGatherV2"GatherV2(1      0@9      0@A      0@I      0@a???7mW?i???<?????Unknown
`
HostGatherV2"
GatherV2_1(1      0@9      0@A      0@I      0@a???7mW?iC?؀ ???Unknown
VHostSum"Sum_2(1      0@9      0@A      0@I      0@a???7mW?i?Љt7???Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a??ᚅHM?i????????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a???7mG?i?O??d???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a???7mG?iT?1@???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a???7mG?iݳ%???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1       @9       @A       @I       @a???7mG?i£???*???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a???7mG?iyj??0???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a???7mG?i01wi?6???Unknown
?HostMatMul",gradient_tape/sequential_72/dense_225/MatMul(1       @9       @A       @I       @a???7mG?i??b??<???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a???7mG?i??NdB???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a@?7??D?i~}??G???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a@?7??D?i^Z?ͣL???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a@?7??D?i>?ٱ?Q???Unknown
?HostMatMul".gradient_tape/sequential_72/dense_225/MatMul_1(1      @9      @A      @I      @a@?7??D?i???V???Unknown
?HostMatMul",gradient_tape/sequential_72/dense_226/MatMul(1      @9      @A      @I      @a@?7??D?i?C6z\???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?$T???A?i??g`???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?$T???A?i?o?d???Unknown
?HostMatMul",gradient_tape/sequential_72/dense_223/MatMul(1      @9      @A      @I      @a?$T???A?iÈ?0i???Unknown
vHost_FusedMatMul"sequential_72/dense_224/Relu(1      @9      @A      @I      @a?$T???A?i"??c?m???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a??ᚅH=?iT??t>q???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a??ᚅH=?i?P`??t???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @a??ᚅH=?i????x???Unknown
v!HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??ᚅH=?i?Ǧ9|???Unknown
?"HostMatMul".gradient_tape/sequential_72/dense_224/MatMul_1(1      @9      @A      @I      @a??ᚅH=?iez?????Unknown
?#HostBiasAddGrad"9gradient_tape/sequential_72/dense_226/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??ᚅH=?iN?-ȋ????Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a???7m7?i??#oy????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???7m7?i?g????Unknown
V&HostCast"Cast(1      @9      @A      @I      @a???7m7?i_k?T????Unknown
\'HostGreater"Greater(1      @9      @A      @I      @a???7m7?i?NdB????Unknown
u(HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a???7m7?i2?
0????Unknown
v)HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a???7m7?ip??????Unknown
~*HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a???7m7?i???X????Unknown
v+HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a???7m7?i&????????Unknown
?,HostBiasAddGrad"9gradient_tape/sequential_72/dense_223/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???7m7?i??Ҧ?????Unknown
?-HostBiasAddGrad"9gradient_tape/sequential_72/dense_224/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???7m7?iܢ?MԠ???Unknown
?.HostBiasAddGrad"9gradient_tape/sequential_72/dense_225/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???7m7?i7????????Unknown
?/HostMatMul".gradient_tape/sequential_72/dense_226/MatMul_1(1      @9      @A      @I      @a???7m7?i?i???????Unknown
y0Host_FusedMatMul"sequential_72/dense_226/BiasAdd(1      @9      @A      @I      @a???7m7?i?L?B?????Unknown
t1HostSigmoid"sequential_72/dense_226/Sigmoid(1      @9      @A      @I      @a???7m7?iH0?銬???Unknown
X2HostCast"Cast_3(1      @9      @A      @I      @a?$T???1?i͚?&?????Unknown
X3HostCast"Cast_4(1      @9      @A      @I      @a?$T???1?iRd?????Unknown
V4HostMean"Mean(1      @9      @A      @I      @a?$T???1?i?oI?!????Unknown
r5HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a?$T???1?i\ځ?S????Unknown
?6HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?$T???1?i?D??????Unknown
z7HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?$T???1?if??X?????Unknown
v8HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?$T???1?i?+??????Unknown
b9HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?$T???1?ip?c?????Unknown
~:HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?$T???1?i???O????Unknown
?;HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?$T???1?izY?M?????Unknown
?<HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?$T???1?i????????Unknown
?=HostReadVariableOp".sequential_72/dense_223/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?$T???1?i?.E??????Unknown
?>HostReadVariableOp".sequential_72/dense_225/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?$T???1?i	?}????Unknown
v?Host_FusedMatMul"sequential_72/dense_225/Relu(1      @9      @A      @I      @a?$T???1?i??BJ????Unknown
t@HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a???7m'?i<?0?????Unknown
XAHostEqual"Equal(1       @9       @A       @I       @a???7m'?i????7????Unknown
sBHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a???7m'?i??&??????Unknown
|CHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a???7m'?iFʡ?%????Unknown
?DHostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1       @9       @A       @I       @a???7m'?i??d?????Unknown
dEHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a???7m'?i???7????Unknown
jFHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a???7m'?iP??????Unknown
vGHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a???7m'?i???? ????Unknown
?HHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a???7m'?i???w????Unknown
}IHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a???7m'?iZt???????Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a???7m'?if?Xe????Unknown
uKHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a???7m'?i?Wy,?????Unknown
wLHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a???7m'?idI??R????Unknown
?MHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a???7m'?i;o??????Unknown
?NHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a???7m'?i?,??@????Unknown
?OHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a???7m'?inez?????Unknown
?PHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a???7m'?i?M.????Unknown
?QHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a???7m'?i?[!?????Unknown
?RHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a???7m'?ix???????Unknown
~SHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a???7m'?i&?PȒ????Unknown
?THostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a???7m'?i??˛	????Unknown
?UHostReadVariableOp"-sequential_72/dense_223/MatMul/ReadVariableOp(1       @9       @A       @I       @a???7m'?i??Fo?????Unknown
?VHostReadVariableOp".sequential_72/dense_224/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a???7m'?i0??B?????Unknown
?WHostReadVariableOp"-sequential_72/dense_224/MatMul/ReadVariableOp(1       @9       @A       @I       @a???7m'?iޫ<n????Unknown
?XHostReadVariableOp".sequential_72/dense_226/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a???7m'?i?????????Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a???7m?icuS?????Unknown
vZHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a???7m?i:?2?[????Unknown
X[HostCast"Cast_5(1      ??9      ??A      ??I      ??a???7m?i?&????Unknown
a\HostIdentity"Identity(1      ??9      ??A      ??I      ??a???7m?i耭??????Unknown?
?]HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a???7m?i??j??????Unknown
T^HostMul"Mul(1      ??9      ??A      ??I      ??a???7m?i?r(dI????Unknown
w_HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a???7m?im???????Unknown
y`HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a???7m?iDd?7?????Unknown
xaHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a???7m?i?`?{????Unknown
?bHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a???7m?i?U7????Unknown
?cHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a???7m?i???t?????Unknown
?dHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a???7m?i?G?ޭ????Unknown
?eHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a???7m?iw?VHi????Unknown
?fHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a???7m?iN9?$????Unknown
?gHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a???7m?i%???????Unknown
?hHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a???7m?i?*???????Unknown
?iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a???7m?iӣL?V????Unknown
?jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a???7m?i?
Y????Unknown
?kHostReluGrad".gradient_tape/sequential_72/dense_223/ReluGrad(1      ??9      ??A      ??I      ??a???7m?i?????????Unknown
?lHostReluGrad".gradient_tape/sequential_72/dense_224/ReluGrad(1      ??9      ??A      ??I      ??a???7m?iX?,?????Unknown
?mHostReluGrad".gradient_tape/sequential_72/dense_225/ReluGrad(1      ??9      ??A      ??I      ??a???7m?i/?B?D????Unknown
?nHostReadVariableOp"-sequential_72/dense_225/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a???7m?i     ???Unknown
?oHostReadVariableOp"-sequential_72/dense_226/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a???7m?in?޴] ???Unknown
YpHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(in?޴] ???Unknown*?n
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a'?l??&??i'?l??&???Unknown?
vHost_FusedMatMul"sequential_72/dense_223/Relu(1      A@9      A@A      A@I      A@a`?1`??i?????????Unknown
dHostDataset"Iterator::Model(1      ;@9      ;@A      ;@I      ;@a?? O	???i???[????Unknown
iHostWriteSummary"WriteSummary(1      ;@9      ;@A      ;@I      ;@a?? O	???iJ????H???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      9@9      9@A      8@I      8@a??)A???i?????????Unknown
?HostMatMul",gradient_tape/sequential_72/dense_224/MatMul(1      3@9      3@A      3@I      3@a{?בz??iU?<%?S???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      1@9      1@A      1@I      1@a`?1`??i????H????Unknown
^HostGatherV2"GatherV2(1      0@9      0@A      0@I      0@a?0?ӈ?i!O	?? ???Unknown
`	HostGatherV2"
GatherV2_1(1      0@9      0@A      0@I      0@a?0?ӈ?i?>??????Unknown
V
HostSum"Sum_2(1      0@9      0@A      0@I      0@a?0?ӈ?i??r?,????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a|???i%?S?<%???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?0??x?iW(n??V???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a?0??x?i?????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a?0??x?i?袋.????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1       @9       @A       @I       @a?0??x?i?H???????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?0??x?i?בz???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?0??x?iQ	?? O???Unknown
?HostMatMul",gradient_tape/sequential_72/dense_225/MatMul(1       @9       @A       @I       @a?0??x?i?i?ƀ???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?0??x?i??&?l????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a??[??u?i?????????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a??[??u?i?? O	???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??[??u?i6Lc?4???Unknown
?HostMatMul".gradient_tape/sequential_72/dense_225/MatMul_1(1      @9      @A      @I      @a??[??u?ia?1`???Unknown
?HostMatMul",gradient_tape/sequential_72/dense_226/MatMul(1      @9      @A      @I      @a??[??u?i?.?袋???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??)A?r?i??k߰???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a??)A?r?i־a?????Unknown
?HostMatMul",gradient_tape/sequential_72/dense_223/MatMul(1      @9      @A      @I      @a??)A?r?i???oX????Unknown
vHost_FusedMatMul"sequential_72/dense_224/Relu(1      @9      @A      @I      @a??)A?r?i O	?? ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a|??o?i?????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a|??o?i^G?u?^???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @a|??o?i}??7?}???Unknown
v HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a|??o?i?????????Unknown
?!HostMatMul".gradient_tape/sequential_72/dense_224/MatMul_1(1      @9      @A      @I      @a|??o?i?????????Unknown
?"HostBiasAddGrad"9gradient_tape/sequential_72/dense_226/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a|??o?i?7?}?????Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?0??h?i?g9?????Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?0??h?i?ƀi???Unknown
V%HostCast"Cast(1      @9      @A      @I      @a?0??h?i%?S?<%???Unknown
\&HostGreater"Greater(1      @9      @A      @I      @a?0??h?i>???>???Unknown
u'HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?0??h?iW(n??V???Unknown
v(HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?0??h?ipX???o???Unknown
~)HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?0??h?i?????????Unknown
v*HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?0??h?i???[????Unknown
?+HostBiasAddGrad"9gradient_tape/sequential_72/dense_223/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?0??h?i?袋.????Unknown
?,HostBiasAddGrad"9gradient_tape/sequential_72/dense_224/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?0??h?i?0?????Unknown
?-HostBiasAddGrad"9gradient_tape/sequential_72/dense_225/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?0??h?i?H???????Unknown
?.HostMatMul".gradient_tape/sequential_72/dense_226/MatMul_1(1      @9      @A      @I      @a?0??h?iyJ?????Unknown
y/Host_FusedMatMul"sequential_72/dense_226/BiasAdd(1      @9      @A      @I      @a?0??h?i?בz???Unknown
t0HostSigmoid"sequential_72/dense_226/Sigmoid(1      @9      @A      @I      @a?0??h?i8?d?M6???Unknown
X1HostCast"Cast_3(1      @9      @A      @I      @a??)A?b?iK????H???Unknown
X2HostCast"Cast_4(1      @9      @A      @I      @a??)A?b?i^???[???Unknown
V3HostMean"Mean(1      @9      @A      @I      @a??)A?b?iq??V(n???Unknown
r4HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a??)A?b?i?i?ƀ???Unknown
?5HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a??)A?b?i?M6?d????Unknown
z6HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??)A?b?i?1`????Unknown
v7HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a??)A?b?i??[?????Unknown
b8HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??)A?b?i?????????Unknown
~9HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a??)A?b?i?????????Unknown
?:HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a??)A?b?i??|????Unknown
?;HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??)A?b?i	?1`???Unknown
?<HostReadVariableOp".sequential_72/dense_223/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??)A?b?i?[?????Unknown
?=HostReadVariableOp".sequential_72/dense_225/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??)A?b?i/n??V(???Unknown
v>Host_FusedMatMul"sequential_72/dense_225/Relu(1      @9      @A      @I      @a??)A?b?iBR?#?:???Unknown
t?HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?0??X?iN?u?^G???Unknown
X@HostEqual"Equal(1       @9       @A       @I       @a?0??X?iZ?<%?S???Unknown
sAHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?0??X?if?1`???Unknown
|BHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?0??X?ir??&?l???Unknown
?CHostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1       @9       @A       @I       @a?0??X?i~J??y???Unknown
dDHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?0??X?i??V(n????Unknown
jEHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?0??X?i?z?ב???Unknown
vFHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?0??X?i??)A????Unknown
?GHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?0??X?i?????????Unknown
}HHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?0??X?i?Bq+????Unknown
`IHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?0??X?i??7?}????Unknown
uJHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?0??X?i?r?,?????Unknown
wKHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?0??X?i?
ŭP????Unknown
?LHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?0??X?iꢋ.?????Unknown
?MHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?0??X?i?:R?#????Unknown
?NHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?0??X?i?0????Unknown
?OHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a?0??X?ik߰????Unknown
?PHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a?0??X?i?1`???Unknown
?QHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?0??X?i&?l??&???Unknown
~RHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?0??X?i233333???Unknown
?SHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?0??X?i>????????Unknown
?THostReadVariableOp"-sequential_72/dense_223/MatMul/ReadVariableOp(1       @9       @A       @I       @a?0??X?iJc?4L???Unknown
?UHostReadVariableOp".sequential_72/dense_224/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?0??X?iV???oX???Unknown
?VHostReadVariableOp"-sequential_72/dense_224/MatMul/ReadVariableOp(1       @9       @A       @I       @a?0??X?ib?M6?d???Unknown
?WHostReadVariableOp".sequential_72/dense_226/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?0??X?in+?Bq???Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?0??H?itwwwww???Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?0??H?iz??7?}???Unknown
XZHostCast"Cast_5(1      ??9      ??A      ??I      ??a?0??H?i?>??????Unknown
a[HostIdentity"Identity(1      ??9      ??A      ??I      ??a?0??H?i?[??????Unknown?
?\HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a?0??H?i??yJ????Unknown
T]HostMul"Mul(1      ??9      ??A      ??I      ??a?0??H?i??g9????Unknown
w^HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?0??H?i?????????Unknown
y_HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?0??H?i??.??????Unknown
x`HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a?0??H?i?בz????Unknown
?aHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?0??H?i?#?:R????Unknown
?bHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?0??H?i?oX??????Unknown
?cHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?0??H?i?????????Unknown
?dHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?0??H?i?|?????Unknown
?eHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a?0??H?i?S?<%????Unknown
?fHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?0??H?iȟ??Y????Unknown
?gHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?0??H?i??H??????Unknown
?hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?0??H?i?7?}?????Unknown
?iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?0??H?iڃ>?????Unknown
?jHostReluGrad".gradient_tape/sequential_72/dense_223/ReluGrad(1      ??9      ??A      ??I      ??a?0??H?i??r?,????Unknown
?kHostReluGrad".gradient_tape/sequential_72/dense_224/ReluGrad(1      ??9      ??A      ??I      ??a?0??H?i?־a????Unknown
?lHostReluGrad".gradient_tape/sequential_72/dense_225/ReluGrad(1      ??9      ??A      ??I      ??a?0??H?i?g9?????Unknown
?mHostReadVariableOp"-sequential_72/dense_225/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?0??H?i?????????Unknown
?nHostReadVariableOp"-sequential_72/dense_226/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?0??H?i?????????Unknown
YoHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU