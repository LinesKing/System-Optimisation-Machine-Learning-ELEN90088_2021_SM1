"?e
BHostIDLE"IDLE1     ??@A     ??@a???????i????????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a?&z߫??iܡ?WG???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?F@9     ?F@A     ?F@I     ?F@aɮ?#^?p?i:![?.i???Unknown
iHostWriteSummary"WriteSummary(1     ?C@9     ?C@A     ?C@I     ?C@a?=?Sm?iG:?_?????Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      A@9      A@A      A@I      A@a???-?i?if?"?????Unknown
zHost_FusedMatMul" sequential_100/dense_312/BiasAdd(1      @@9      @@A      @@I      @@aLY??*h?i????#????Unknown
wHost_FusedMatMul"sequential_100/dense_310/Relu(1      6@9      6@A      6@I      6@ad}?w?`?i<lwծ????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@a?o??5^?i?kA??????Unknown
?	HostReadVariableOp"/sequential_100/dense_311/BiasAdd/ReadVariableOp(1      3@9      3@A      3@I      3@a
?2?2?\?iI???????Unknown
g
HostStridedSlice"strided_slice(1      1@9      1@A      1@I      1@a???-?Y?i??y ?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      0@9      0@A      0@I      0@aLY??*X?i8?5?????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      ,@9      ,@A      ,@I      ,@a"?2?%U?ilѨHZ	???Unknown
?HostMatMul"-gradient_tape/sequential_100/dense_312/MatMul(1      $@9      $@A      $@I      $@a?o??5N?iH?V????Unknown
`HostGatherV2"
GatherV2_1(1      "@9      "@A      "@I      "@au?e80K?i???????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@au?e80K?i:*nh???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@au?e80K?i?8?,%???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@au?e80K?i,7F??+???Unknown
?HostMatMul"/gradient_tape/sequential_100/dense_311/MatMul_1(1      "@9      "@A      "@I      "@au?e80K?i?PT?2???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @aLY??*H?i???8???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @aLY??*H?iѶ?'?>???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @aLY??*H?i??y2?D???Unknown?
dHostDataset"Iterator::Model(1      ;@9      ;@A      @I      @a"?2?%E?i?6ڻJ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a"?2?%E?iO?:EIO???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a"?2?%E?iКΌT???Unknown
?HostMatMul"-gradient_tape/sequential_100/dense_310/MatMul(1      @9      @A      @I      @a"?2?%E?i??W?Y???Unknown
wHost_FusedMatMul"sequential_100/dense_311/Relu(1      @9      @A      @I      @a"?2?%E?iki[?_???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?B?% B?i??d??c???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?B?% B?i6n?h???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?o??5>?i?? x?k???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?o??5>?i?5???o???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?o??5>?i׵??as???Unknown
V HostSum"Sum_2(1      @9      @A      @I      @a?o??5>?i?58$w???Unknown
?!HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?o??5>?i?????z???Unknown
?"HostBiasAddGrad":gradient_tape/sequential_100/dense_310/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?o??5>?i?5??~???Unknown
?#HostBiasAddGrad":gradient_tape/sequential_100/dense_311/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?o??5>?i??O?k????Unknown
?$HostMatMul"-gradient_tape/sequential_100/dense_311/MatMul(1      @9      @A      @I      @a?o??5>?i}5'.????Unknown
?%HostReadVariableOp".sequential_100/dense_312/MatMul/ReadVariableOp(1      @9      @A      @I      @a?o??5>?ik????????Unknown
V&HostMean"Mean(1      @9      @A      @I      @aLY??*8?i?N??????Unknown
s'HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @aLY??*8?i??k??????Unknown
r(HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @aLY??*8?i?ǽ?????Unknown
v)HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aLY??*8?i?#??????Unknown
~*HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aLY??*8?i"?~??????Unknown
v+HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aLY??*8?i?N???????Unknown
v,HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aLY??*8?i8?5??????Unknown
?-HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @aLY??*8?iÁ?? ????Unknown
~.HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @aLY??*8?iN??????Unknown
?/HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aLY??*8?iٴH?????Unknown
?0Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @aLY??*8?idN??????Unknown
?1HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aLY??*8?i????????Unknown
?2HostBiasAddGrad":gradient_tape/sequential_100/dense_312/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aLY??*8?iz?[?
????Unknown
?3HostMatMul"/gradient_tape/sequential_100/dense_312/MatMul_1(1      @9      @A      @I      @aLY??*8?i??????Unknown
?4HostReadVariableOp".sequential_100/dense_310/MatMul/ReadVariableOp(1      @9      @A      @I      @aLY??*8?i???????Unknown
?5HostReadVariableOp".sequential_100/dense_311/MatMul/ReadVariableOp(1      @9      @A      @I      @aLY??*8?iNn????Unknown
t6HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?B?% 2?iCs?R????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_1(1      @9      @A      @I      @a?B?% 2?ik?w?????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a?B?% 2?i?g|??????Unknown
X9HostCast"Cast_3(1      @9      @A      @I      @a?B?% 2?i??????Unknown
\:HostGreater"Greater(1      @9      @A      @I      @a?B?% 2?i?ͅ?X????Unknown
u;HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?B?% 2?i???????Unknown
|<HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a?B?% 2?i34???????Unknown
j=HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?B?% 2?i[??#????Unknown
?>HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?B?% 2?i????^????Unknown
v?HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?B?% 2?i?M?+?????Unknown
}@HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      @9      @A      @I      @a?B?% 2?i? ???????Unknown
?AHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?B?% 2?i???3#????Unknown
?BHostReluGrad"/gradient_tape/sequential_100/dense_311/ReluGrad(1      @9      @A      @I      @a?B?% 2?i#g??d????Unknown
VCHostCast"Cast(1       @9       @A       @I       @aLY??*(?i?3Y??????Unknown
XDHostEqual"Equal(1       @9       @A       @I       @aLY??*(?i? ?f????Unknown
dEHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @aLY??*(?iuʹ??????Unknown
vFHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aLY??*(?i;?b?h????Unknown
zGHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @aLY??*(?ig??????Unknown
`HHostDivNoNan"
div_no_nan(1       @9       @A       @I       @aLY??*(?i?3??j????Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aLY??*(?i? l??????Unknown
bJHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @aLY??*(?iS??l????Unknown
wKHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aLY??*(?i????????Unknown
?LHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @aLY??*(?i?fu?n????Unknown
?MHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @aLY??*(?i?3#??????Unknown
?NHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @aLY??*(?ik ??p????Unknown
?OHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @aLY??*(?i1?~??????Unknown
?PHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @aLY??*(?i??,?r????Unknown
?QHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @aLY??*(?i?f???????Unknown
?RHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @aLY??*(?i?3??t????Unknown
?SHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @aLY??*(?iI 6??????Unknown
~THostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @aLY??*(?i???v????Unknown
?UHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aLY??*(?iՙ???????Unknown
uVHostSigmoid" sequential_100/dense_312/Sigmoid(1       @9       @A       @I       @aLY??*(?i?f??x????Unknown
XWHostCast"Cast_4(1      ??9      ??A      ??I      ??aLY??*?i?L?n9????Unknown
XXHostCast"Cast_5(1      ??9      ??A      ??I      ??aLY??*?ia3???????Unknown
aYHostIdentity"Identity(1      ??9      ??A      ??I      ??aLY??*?i?Dq?????Unknown?
TZHostMul"Mul(1      ??9      ??A      ??I      ??aLY??*?i' ??z????Unknown
w[HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aLY??*?i???s;????Unknown
y\HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aLY??*?i??H??????Unknown
x]HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??aLY??*?iP??v?????Unknown
?^HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??aLY??*?i????|????Unknown
?_HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aLY??*?i?My=????Unknown
?`HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aLY??*?iyf???????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??aLY??*?i?L?{?????Unknown
?bHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??aLY??*?i?3R?~????Unknown
?cHostReluGrad"/gradient_tape/sequential_100/dense_310/ReluGrad(1      ??9      ??A      ??I      ??aLY??*?i??~?????Unknown
?dHostReadVariableOp"/sequential_100/dense_310/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aLY??*?i     ???Unknown
?eHostReadVariableOp"/sequential_100/dense_312/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aLY??*?i3s?@` ???Unknown
ifHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i3s?@` ???Unknown
WgHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i3s?@` ???Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i3s?@` ???Unknown*?e
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a?]? ????i?]? ?????Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?F@9     ?F@A     ?F@I     ?F@a??|?4??i??x???Unknown
iHostWriteSummary"WriteSummary(1     ?C@9     ?C@A     ?C@I     ?C@a??VCӝ?i?_~??????Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      A@9      A@A      A@I      A@a?fR?a ??i1󘠨????Unknown
zHost_FusedMatMul" sequential_100/dense_312/BiasAdd(1      @@9      @@A      @@I      @@af????x??i??fEo????Unknown
wHost_FusedMatMul"sequential_100/dense_310/Relu(1      6@9      6@A      6@I      6@aʭ)Ӑ?i=?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@a??$?	???i???c????Unknown
?HostReadVariableOp"/sequential_100/dense_311/BiasAdd/ReadVariableOp(1      3@9      3@A      3@I      3@a?E?v|??i)ю?????Unknown
g	HostStridedSlice"strided_slice(1      1@9      1@A      1@I      1@a?fR?a ??i?w?k???Unknown
?
HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      0@9      0@A      0@I      0@af????x??i???Ɇ????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      ,@9      ,@A      ,@I      ,@ay??i??i???-#???Unknown
?HostMatMul"-gradient_tape/sequential_100/dense_312/MatMul(1      $@9      $@A      $@I      $@a??$?	?~?in@?[`???Unknown
`HostGatherV2"
GatherV2_1(1      "@9      "@A      "@I      "@aR??,??{?iw?k????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@aR??,??{?i??Ё{????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aR??,??{?iu?*`????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aR??,??{?i"??>?<???Unknown
?HostMatMul"/gradient_tape/sequential_100/dense_311/MatMul_1(1      "@9      "@A      "@I      "@aR??,??{?iϊ??s???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @af????xx?i??Ɯ????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @af????xx?i?fEo?????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @af????xx?i??x????Unknown?
dHostDataset"Iterator::Model(1      ;@9      ;@A      @I      @ay??iu?i?ԅ?S1???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @ay??iu?i?Ԓ '\???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @ay??iu?i/՟t?????Unknown
?HostMatMul"-gradient_tape/sequential_100/dense_310/MatMul(1      @9      @A      @I      @ay??iu?i`լ?ͱ???Unknown
wHost_FusedMatMul"sequential_100/dense_311/Relu(1      @9      @A      @I      @ay??iu?i?չ\?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?9Is?Zr?ih??V???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?9Is?Zr?iw???&???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??$?	?n?i,G??D???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??$?	?n?i?C?9c???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??$?	?n?i?h??Ё???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a??$?	?n?iK??h????Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??$?	?n?i ?G?????Unknown
?!HostBiasAddGrad":gradient_tape/sequential_100/dense_310/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??$?	?n?i???????Unknown
?"HostBiasAddGrad":gradient_tape/sequential_100/dense_311/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??$?	?n?ij??-????Unknown
?#HostMatMul"-gradient_tape/sequential_100/dense_311/MatMul(1      @9      @A      @I      @a??$?	?n?i ?(????Unknown
?$HostReadVariableOp".sequential_100/dense_312/MatMul/ReadVariableOp(1      @9      @A      @I      @a??$?	?n?i?DH2[9???Unknown
V%HostMean"Mean(1      @9      @A      @I      @af????xh?i????Q???Unknown
s&HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @af????xh?i²{?Lj???Unknown
r'HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @af????xh?i?i?ł???Unknown
v(HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @af????xh?i? ??>????Unknown
~)HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @af????xh?i??HY?????Unknown
v*HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @af????xh?i???-0????Unknown
v+HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @af????xh?i?E|?????Unknown
?,HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @af????xh?i???!????Unknown
~-HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @af????xh?i????????Unknown
?.HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @af????xh?izjI?.???Unknown
?/Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @af????xh?iq!?T?F???Unknown
?0HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @af????xh?ih?|)_???Unknown
?1HostBiasAddGrad":gradient_tape/sequential_100/dense_312/BiasAdd/BiasAddGrad(1      @9      @A      @I      @af????xh?i_??}w???Unknown
?2HostMatMul"/gradient_tape/sequential_100/dense_312/MatMul_1(1      @9      @A      @I      @af????xh?iVF???????Unknown
?3HostReadVariableOp".sequential_100/dense_310/MatMul/ReadVariableOp(1      @9      @A      @I      @af????xh?iM?I?o????Unknown
?4HostReadVariableOp".sequential_100/dense_311/MatMul/ReadVariableOp(1      @9      @A      @I      @af????xh?iD??{?????Unknown
t5HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?9Is?Zb?i~?VC????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_1(1      @9      @A      @I      @a?9Is?Zb?i?Fʺ?????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a?9Is?Zb?i??=Z?????Unknown
X8HostCast"Cast_3(1      @9      @A      @I      @a?9Is?Zb?i,ٰ?R
???Unknown
\9HostGreater"Greater(1      @9      @A      @I      @a?9Is?Zb?if"$?????Unknown
u:HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?9Is?Zb?i?k?8/???Unknown
|;HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a?9Is?Zb?iڴ
?bA???Unknown
j<HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?9Is?Zb?i?}w?S???Unknown
?=HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?9Is?Zb?iNG?f???Unknown
v>HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?9Is?Zb?i??d?rx???Unknown
}?HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      @9      @A      @I      @a?9Is?Zb?i???U͊???Unknown
?@HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?9Is?Zb?i?"K?'????Unknown
?AHostReluGrad"/gradient_tape/sequential_100/dense_311/ReluGrad(1      @9      @A      @I      @a?9Is?Zb?i6l???????Unknown
VBHostCast"Cast(1       @9       @A       @I       @af????xX?i?G??????Unknown
XCHostEqual"Equal(1       @9       @A       @I       @af????xX?i.#Xi?????Unknown
dDHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @af????xX?i????7????Unknown
vEHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @af????xX?i&??=t????Unknown
zFHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @af????xX?i??>??????Unknown
`GHostDivNoNan"
div_no_nan(1       @9       @A       @I       @af????xX?i???????Unknown
uHHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @af????xX?i?l?|)???Unknown
bIHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @af????xX?iH%?e???Unknown
wJHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @af????xX?i?#rQ????Unknown
?KHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @af????xX?i????)???Unknown
?LHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @af????xX?i??&6???Unknown
?MHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @af????xX?i?X?WB???Unknown
?NHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @af????xX?i?????N???Unknown
?OHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @af????xX?i?l?d?Z???Unknown
?PHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @af????xX?izH??g???Unknown
?QHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @af????xX?i?#?9Is???Unknown
?RHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @af????xX?ir?أ????Unknown
~SHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @af????xX?i??%???Unknown
?THostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @af????xX?ij?rx?????Unknown
uUHostSigmoid" sequential_100/dense_312/Sigmoid(1       @9       @A       @I       @af????xX?i摿?:????Unknown
XVHostCast"Cast_4(1      ??9      ??A      ??I      ??af????xH?i???Y????Unknown
XWHostCast"Cast_5(1      ??9      ??A      ??I      ??af????xH?ibmMw????Unknown
aXHostIdentity"Identity(1      ??9      ??A      ??I      ??af????xH?i ?2??????Unknown?
TYHostMul"Mul(1      ??9      ??A      ??I      ??af????xH?i?HY??????Unknown
wZHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??af????xH?i????????Unknown
y[HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??af????xH?iZ$?!?????Unknown
x\HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??af????xH?i??V????Unknown
?]HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??af????xH?i????,????Unknown
?^HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??af????xH?i?m?J????Unknown
?_HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??af????xH?iR???h????Unknown
?`HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??af????xH?iIf+?????Unknown
?aHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??af????xH?iζ?`?????Unknown
?bHostReluGrad"/gradient_tape/sequential_100/dense_310/ReluGrad(1      ??9      ??A      ??I      ??af????xH?i?$???????Unknown
?cHostReadVariableOp"/sequential_100/dense_310/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??af????xH?iJ????????Unknown
?dHostReadVariableOp"/sequential_100/dense_312/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??af????xH?i     ???Unknown
ieHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i     ???Unknown
WfHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i     ???Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown2CPU