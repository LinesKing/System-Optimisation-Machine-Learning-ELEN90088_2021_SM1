"?h
BHostIDLE"IDLE1    ?}?@A    ?}?@aK???%???iK???%????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     0?@9     0?@A     0?@I     0?@a:^\?,??iQ)????Unknown?
iHostWriteSummary"WriteSummary(1      @@9      @@A      @@I      @@a?g?[]i?i?`9???Unknown?
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      ?@9      ?@A      ?@I      ?@a??k q?h?i?#	їQ???Unknown
yHost_FusedMatMul"sequential_54/dense_170/BiasAdd(1      =@9      =@A      =@I      =@ax\uB??f?i@?Kl?h???Unknown
vHost_FusedMatMul"sequential_54/dense_168/Relu(1      ;@9      ;@A      ;@I      ;@a)?~??fe?i,?1?}???Unknown
tHostSigmoid"sequential_54/dense_170/Sigmoid(1      8@9      8@A      8@I      8@a?C??c?ip??6????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      5@I      5@a<??JD?`?iA{?????Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(1      3@9      3@A      3@I      3@a?UJ?^?i6??鵰???Unknown
?
HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      0@9      0@A      0@I      0@a?g?[]Y?i??~?d????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      0@9      0@A      0@I      0@a?g?[]Y?i:MnE????Unknown
gHostStridedSlice"strided_slice(1      0@9      0@A      0@I      0@a?g?[]Y?i? ^??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      .@9      .@A      .@I      .@a??p!??W?i?n??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1      ,@9      ,@A      ,@I      ,@aP$zc?1V?iv???????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      *@9      *@A      *@I      *@a???ڛT?i?7?{????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      (@9      (@A      (@I      (@a?C??S?i??f~????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      &@9      &@A      &@I      &@adӖ)/pQ?i????G
???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      &@9      &@A      &@I      &@adӖ)/pQ?ih???????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      $@9      $@A      $@I      $@a)?@ײ?O?i?eF?????Unknown
dHostDataset"Iterator::Model(1      =@9      =@A      $@I      $@a)?@ײ?O?i?5??"???Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a)?@ײ?O?i??3?*???Unknown
[HostAddV2"Adam/add(1      "@9      "@A      "@I      "@a??S[?L?i?ڈu?1???Unknown
?HostMatMul",gradient_tape/sequential_54/dense_170/MatMul(1      "@9      "@A      "@I      "@a??S[?L?i??_?9???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?g?[]I?i??Wc????Unknown?
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?g?[]I?ircOe?E???Unknown
?HostMatMul",gradient_tape/sequential_54/dense_169/MatMul(1       @9       @A       @I       @a?g?[]I?i3=G?L???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aP$zc?1F?i?`(?Q???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @aP$zc?1F?iE?x?*W???Unknown
?HostMatMul",gradient_tape/sequential_54/dense_168/MatMul(1      @9      @A      @I      @aP$zc?1F?i?ؑ ?\???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?C??C?i?ˁxa???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?C??C?ip?:f???Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?C??C?i?????j???Unknown
?!HostMatMul".gradient_tape/sequential_54/dense_169/MatMul_1(1      @9      @A      @I      @a?C??C?ify?o???Unknown
?"HostBiasAddGrad"9gradient_tape/sequential_54/dense_170/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?C??C?icI??~t???Unknown
v#Host_FusedMatMul"sequential_54/dense_169/Relu(1      @9      @A      @I      @a?C??C?i?,?@y???Unknown
?$HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a)?@ײ???i?H?6}???Unknown
?%HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a)?@ײ???i???4-????Unknown
?&HostBiasAddGrad"9gradient_tape/sequential_54/dense_168/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a)?@ײ???i????#????Unknown
Y'HostPow"Adam/Pow(1      @9      @A      @I      @a?g?[]9?i??yvO????Unknown
\(HostGreater"Greater(1      @9      @A      @I      @a?g?[]9?i???!{????Unknown
V)HostSum"Sum_2(1      @9      @A      @I      @a?g?[]9?i??qͦ????Unknown
b*HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?g?[]9?i???xґ???Unknown
?+HostBiasAddGrad"9gradient_tape/sequential_54/dense_169/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?g?[]9?id?i$?????Unknown
?,HostReadVariableOp".sequential_54/dense_168/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?g?[]9?iEr??)????Unknown
?-HostReadVariableOp"-sequential_54/dense_170/MatMul/ReadVariableOp(1      @9      @A      @I      @a?g?[]9?i&_a{U????Unknown
t.HostReadVariableOp"Adam/Cast/ReadVariableOp(1      @9      @A      @I      @a?C??3?i?P?;?????Unknown
]/HostCast"Adam/Cast_1(1      @9      @A      @I      @a?C??3?ivB??????Unknown
v0HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a?C??3?i48?w????Unknown
[1HostPow"
Adam/Pow_1(1      @9      @A      @I      @a?C??3?i?%?}ؤ???Unknown
o2HostReadVariableOp"Adam/ReadVariableOp(1      @9      @A      @I      @a?C??3?inr>9????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?C??3?i	??????Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?C??3?i?????????Unknown
X5HostCast"Cast_3(1      @9      @A      @I      @a?C??3?if?H?[????Unknown
V6HostMean"Mean(1      @9      @A      @I      @a?C??3?i??@?????Unknown
j7HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?C??3?i?ς????Unknown
~8HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?C??3?i^??}????Unknown
v9HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?C??3?i???޷???Unknown
v:HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?C??3?i??YC?????Unknown
?;HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a?C??3?iV???????Unknown
u<HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a?C??3?i???? ????Unknown
?=HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?C??3?i?y0?a????Unknown
?>HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?C??3?iNk?E?????Unknown
??Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?C??3?i?\j#????Unknown
?@Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a?C??3?i?Nǃ????Unknown
?AHostMatMul".gradient_tape/sequential_54/dense_170/MatMul_1(1      @9      @A      @I      @a?C??3?iF@???????Unknown
?BHostReadVariableOp"-sequential_54/dense_168/MatMul/ReadVariableOp(1      @9      @A      @I      @a?C??3?i?1AHE????Unknown
~CHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1       @9       @A       @I       @a?g?[])?i^(??????Unknown
eDHostAddN"Adam/gradients/AddN(1       @9       @A       @I       @a?g?[])?i???p????Unknown
tEHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?g?[])?i>{?????Unknown
VFHostCast"Cast(1       @9       @A       @I       @a?g?[])?i?9??????Unknown
XGHostCast"Cast_5(1       @9       @A       @I       @a?g?[])?i?t2????Unknown
XHHostEqual"Equal(1       @9       @A       @I       @a?g?[])?i???J?????Unknown
aIHostIdentity"Identity(1       @9       @A       @I       @a?g?[])?i??r ^????Unknown?
rJHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?g?[])?in?0??????Unknown
vKHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?g?[])?i???ˉ????Unknown
?LHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a?g?[])?iNҬ?????Unknown
zMHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a?g?[])?i??jw?????Unknown
vNHostSum"%binary_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @a?g?[])?i.?(MK????Unknown
`OHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?g?[])?i???"?????Unknown
wPHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?g?[])?i???v????Unknown
?QHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?g?[])?i~?b?????Unknown
~RHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?g?[])?i?? ??????Unknown
?SHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?g?[])?i^??y8????Unknown
?THostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?g?[])?i΅?O?????Unknown
?UHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a?g?[])?i>|Z%d????Unknown
~VHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?g?[])?i?r??????Unknown
?WHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?g?[])?ii?Џ????Unknown
?XHostReluGrad".gradient_tape/sequential_54/dense_168/ReluGrad(1       @9       @A       @I       @a?g?[])?i?_??%????Unknown
?YHostReluGrad".gradient_tape/sequential_54/dense_169/ReluGrad(1       @9       @A       @I       @a?g?[])?i?UR|?????Unknown
?ZHostReadVariableOp"-sequential_54/dense_169/MatMul/ReadVariableOp(1       @9       @A       @I       @a?g?[])?inLRQ????Unknown
v[HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??a?g?[]?i?G?<????Unknown
v\HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?g?[]?i?B?'?????Unknown
v]HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?g?[]?i>??????Unknown
X^HostCast"Cast_4(1      ??9      ??A      ??I      ??a?g?[]?iN9??|????Unknown
T_HostMul"Mul(1      ??9      ??A      ??I      ??a?g?[]?i?4k?G????Unknown
}`HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?g?[]?i?/J?????Unknown
waHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?g?[]?i?*)??????Unknown
ybHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?g?[]?i.&??????Unknown
xcHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a?g?[]?if!??s????Unknown
?dHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?g?[]?i??~>????Unknown
?eHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?g?[]?i??i	????Unknown
?fHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?g?[]?i?T?????Unknown
?gHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a?g?[]?iFc??????Unknown
?hHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?g?[]?i~	B*j????Unknown
?iHostReadVariableOp".sequential_54/dense_169/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?g?[]?i?!5????Unknown
?jHostReadVariableOp".sequential_54/dense_170/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?g?[]?i?????????Unknown
WkHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i?????????Unknown
[lHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
YmHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown
[nHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown*?h
uHostFlushSummaryWriter"FlushSummaryWriter(1     0?@9     0?@A     0?@I     0?@a???????i????????Unknown?
iHostWriteSummary"WriteSummary(1      @@9      @@A      @@I      @@a|.?????i+6?W?????Unknown?
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      ?@9      ?@A      ?@I      ?@ap???? ??i?<G?h???Unknown
yHost_FusedMatMul"sequential_54/dense_170/BiasAdd(1      =@9      =@A      =@I      =@aX?y?҄??iA??????Unknown
vHost_FusedMatMul"sequential_54/dense_168/Relu(1      ;@9      ;@A      ;@I      ;@aA?V????i˞:????Unknown
tHostSigmoid"sequential_54/dense_170/Sigmoid(1      8@9      8@A      8@I      8@a???ϑ?iԲ?N?C???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      5@I      5@a?aܯK*??i\$g}/????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      3@9      3@A      3@I      3@aá?=u2??i?~]R?0???Unknown
?	HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      0@9      0@A      0@I      0@a|.?????i?6? ?????Unknown
?
HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      0@9      0@A      0@I      0@a|.?????i?????????Unknown
gHostStridedSlice"strided_slice(1      0@9      0@A      0@I      0@a|.?????i??7??M???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      .@9      .@A      .@I      .@ad!Y?B??i{ӛ??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1      ,@9      ,@A      ,@I      ,@aLA??Ƅ?i?tS????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      *@9      *@A      *@I      *@a5a???J??i??<G???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      (@9      (@A      (@I      (@a???ρ?i	m5x????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      &@9      &@A      &@I      &@a?tS??i????????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      &@9      &@A      &@I      &@a?tS??i???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      $@9      $@A      $@I      $@aہ?v`?}?i???mL???Unknown
dHostDataset"Iterator::Model(1      =@9      =@A      $@I      $@aہ?v`?}?i???ʇ???Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@aہ?v`?}?ij?S'????Unknown
[HostAddV2"Adam/add(1      "@9      "@A      "@I      "@a??s??z?i?Q?g?????Unknown
?HostMatMul",gradient_tape/sequential_54/dense_170/MatMul(1      "@9      "@A      "@I      "@a??s??z?i#9?{.???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a|.???w?i&??~]???Unknown?
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a|.???w?i)?3J?????Unknown
?HostMatMul",gradient_tape/sequential_54/dense_169/MatMul(1       @9       @A       @I       @a|.???w?i,MX?y????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aLA???t?i??k????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @aLA???t?i2??%????Unknown
?HostMatMul",gradient_tape/sequential_54/dense_168/MatMul(1      @9      @A      @I      @aLA???t?i???"9???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a????q?i?s??\???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a????q?i?H??^????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a????q?i??)?????Unknown
? HostMatMul".gradient_tape/sequential_54/dense_169/MatMul_1(1      @9      @A      @I      @a????q?i?҄?????Unknown
?!HostBiasAddGrad"9gradient_tape/sequential_54/dense_170/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????q?i??"9????Unknown
v"Host_FusedMatMul"sequential_54/dense_169/Relu(1      @9      @A      @I      @a????q?i?\;0????Unknown
?#HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @aہ?v`?m?iC???,???Unknown
?$HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aہ?v`?m?i??(?3J???Unknown
?%HostBiasAddGrad"9gradient_tape/sequential_54/dense_168/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aہ?v`?m?iG??Q?g???Unknown
Y&HostPow"Adam/Pow(1      @9      @A      @I      @a|.???g?iH?1????Unknown
\'HostGreater"Greater(1      @9      @A      @I      @a|.???g?iI?ø_????Unknown
V(HostSum"Sum_2(1      @9      @A      @I      @a|.???g?iJVl????Unknown
b)HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a|.???g?iKA??????Unknown
?*HostBiasAddGrad"9gradient_tape/sequential_54/dense_169/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a|.???g?iLozӛ????Unknown
?+HostReadVariableOp".sequential_54/dense_168/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a|.???g?iM??Z????Unknown
?,HostReadVariableOp"-sequential_54/dense_170/MatMul/ReadVariableOp(1      @9      @A      @I      @a|.???g?iN˞:???Unknown
t-HostReadVariableOp"Adam/Cast/ReadVariableOp(1      @9      @A      @I      @a????a?i?mLA????Unknown
].HostCast"Adam/Cast_1(1      @9      @A      @I      @a????a?iP?G?1???Unknown
v/HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a????a?iѲ?N?C???Unknown
[0HostPow"
Adam/Pow_1(1      @9      @A      @I      @a????a?iRUUUUU???Unknown
o1HostReadVariableOp"Adam/ReadVariableOp(1      @9      @A      @I      @a????a?i??\$g???Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a????a?iT??b?x???Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a????a?i?<^i???Unknown
X4HostCast"Cast_3(1      @9      @A      @I      @a????a?iV?p?????Unknown
V5HostMean"Mean(1      @9      @A      @I      @a????a?iׁ?v`????Unknown
j6HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a????a?iX$g}/????Unknown
~7HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a????a?i????????Unknown
v8HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a????a?iZi?????Unknown
v9HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a????a?i?p??????Unknown
?:HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a????a?i\??k???Unknown
u;HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a????a?i?P˞:???Unknown
?<HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a????a?i^?x?	+???Unknown
?=HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a????a?iߕ&??<???Unknown
?>Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a????a?i`8Բ?N???Unknown
??Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a????a?i?ځ?v`???Unknown
?@HostMatMul".gradient_tape/sequential_54/dense_170/MatMul_1(1      @9      @A      @I      @a????a?ib}/?Er???Unknown
?AHostReadVariableOp"-sequential_54/dense_168/MatMul/ReadVariableOp(1      @9      @A      @I      @a????a?i???????Unknown
~BHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1       @9       @A       @I       @a|.???W?i?6? ?????Unknown
eCHostAddN"Adam/gradients/AddN(1       @9       @A       @I       @a|.???W?i?Mozӛ???Unknown
tDHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a|.???W?i?d8Բ????Unknown
VEHostCast"Cast(1       @9       @A       @I       @a|.???W?i?{.?????Unknown
XFHostCast"Cast_5(1       @9       @A       @I       @a|.???W?i??ʇq????Unknown
XGHostEqual"Equal(1       @9       @A       @I       @a|.???W?i驓?P????Unknown
aHHostIdentity"Identity(1       @9       @A       @I       @a|.???W?i??\;0????Unknown?
rIHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a|.???W?i??%?????Unknown
vJHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a|.???W?i?????????Unknown
?KHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a|.???W?i??H?????Unknown
zLHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a|.???W?i???????Unknown
vMHostSum"%binary_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @a|.???W?i?3J?????Unknown
`NHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a|.???W?i?JVl???Unknown
wOHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a|.???W?i?aܯK*???Unknown
?PHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a|.???W?i?x?	+6???Unknown
~QHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a|.???W?i??nc
B???Unknown
?RHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a|.???W?i??7??M???Unknown
?SHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a|.???W?i?? ?Y???Unknown
?THostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a|.???W?i???p?e???Unknown
~UHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a|.???W?i???ʇq???Unknown
?VHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a|.???W?i?\$g}???Unknown
?WHostReluGrad".gradient_tape/sequential_54/dense_168/ReluGrad(1       @9       @A       @I       @a|.???W?i?%~F????Unknown
?XHostReluGrad".gradient_tape/sequential_54/dense_169/ReluGrad(1       @9       @A       @I       @a|.???W?i?0??%????Unknown
?YHostReadVariableOp"-sequential_54/dense_169/MatMul/ReadVariableOp(1       @9       @A       @I       @a|.???W?i?G?1????Unknown
vZHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??a|.???G?i{ӛ??????Unknown
v[HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a|.???G?i?^???????Unknown
v\HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a|.???G?i{?d8Բ???Unknown
X]HostCast"Cast_4(1      ??9      ??A      ??I      ??a|.???G?i?uI?ø???Unknown
T^HostMul"Mul(1      ??9      ??A      ??I      ??a|.???G?i{.??????Unknown
}_HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a|.???G?i????????Unknown
w`HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a|.???G?i{???????Unknown
yaHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a|.???G?i??ۘ?????Unknown
xbHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a|.???G?i{/?Er????Unknown
?cHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a|.???G?i????a????Unknown
?dHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a|.???G?i{F??Q????Unknown
?eHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a|.???G?i??mLA????Unknown
?fHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a|.???G?i{]R?0????Unknown
?gHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a|.???G?i??6? ????Unknown
?hHostReadVariableOp".sequential_54/dense_169/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a|.???G?i{tS????Unknown
?iHostReadVariableOp".sequential_54/dense_170/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a|.???G?i?????????Unknown
WjHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i?????????Unknown
[kHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
YlHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown
[mHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown2CPU