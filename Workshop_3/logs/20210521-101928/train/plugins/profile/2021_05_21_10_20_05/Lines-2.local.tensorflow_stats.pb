"?O
BHostIDLE"IDLE1     ??@A     ??@a?w?W????i?w?W?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     Ѓ@9     Ѓ@A     Ѓ@I     Ѓ@a??XH+۳?i???<???Unknown?
`HostGatherV2"
GatherV2_1(1      @@9      @@A      @@I      @@a???	p?iFC"?\???Unknown
dHostDataset"Iterator::Model(1      :@9      :@A      :@I      :@a?;?>?j?i??`s"v???Unknown
iHostWriteSummary"WriteSummary(1      9@9      9@A      9@I      9@a"?u?i?i\N?0????Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A      2@I      2@aV?5?%
b?i??:????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      1@9      1@A      1@I      1@a?Dd?	a?iG?gFD????Unknown
tHost_FusedMatMul"sequential_9/dense_28/Relu(1      1@9      1@A      1@I      1@a?Dd?	a?i????M????Unknown
?	HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      .@9      .@A      .@I      .@a??Y??^?i}O?PV????Unknown
?
HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      *@9      *@A      *@I      *@a?;?>?Z?i???]????Unknown
tHost_FusedMatMul"sequential_9/dense_29/Relu(1      *@9      *@A      *@I      *@a?;?>?Z?i????d????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      (@9      (@A      (@I      (@as?G??X?i???k????Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_30/MatMul(1      &@9      &@A      &@I      &@a???fV?i??q???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a?A?WFT?i??B?w???Unknown
wHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      $@9      $@A      $@I      $@a?A?WFT?i:?n6}???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      ,@9      ,@A       @I       @a???	P?i???????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a???	P?i?x?;?'???Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_29/MatMul(1       @9       @A       @I       @a???	P?i!?
??/???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a1????L?i?%C??6???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a1????L?iie{??=???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a1????L?i????D???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a1????L?i??놚K???Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_28/MatMul(1      @9      @A      @I      @a1????L?iU$$y?R???Unknown
?HostMatMul",gradient_tape/sequential_9/dense_29/MatMul_1(1      @9      @A      @I      @a1????L?i?c\k?Y???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_9/dense_30/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a1????L?i???]?`???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @as?G??H?i??{??f???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @as?G??H?i??b!?l???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?A?WFD?i?????q???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?A?WFD?i1??Ĳv???Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?A?WFD?i?t$??{???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?A?WFD?i?X?g?????Unknown
? HostBiasAddGrad"7gradient_tape/sequential_9/dense_29/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?A?WFD?i!=P9?????Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???	@?i???z?????Unknown
V"HostMean"Mean(1      @9      @A      @I      @a???	@?io?ٻ?????Unknown
V#HostSum"Sum_2(1      @9      @A      @I      @a???	@?ia??????Unknown
?$HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a???	@?i?c>ĕ???Unknown
z%HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a???	@?idΧƙ???Unknown
v&HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a???	@?i???ȝ???Unknown
v'HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a???	@?i?;1ˡ???Unknown
?(HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a???	@?iY?uCͥ???Unknown
?)HostBiasAddGrad"7gradient_tape/sequential_9/dense_28/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???	@?i ???ϩ???Unknown
?*HostReadVariableOp",sequential_9/dense_28/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???	@?i?_??ѭ???Unknown
r+HostSigmoid"sequential_9/dense_30/Sigmoid(1      @9      @A      @I      @a???	@?iNDԱ???Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @as?G??8?iK?7?մ???Unknown
\-HostGreater"Greater(1      @9      @A      @I      @as?G??8?iH(+i׷???Unknown
s.HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @as?G??8?iE?ٺ???Unknown
u/HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @as?G??8?iB:?ڽ???Unknown
d0HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @as?G??8?i??|?????Unknown
`1HostDivNoNan"
div_no_nan(1      @9      @A      @I      @as?G??8?i<L?,?????Unknown
b2HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @as?G??8?i9????????Unknown
?3HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @as?G??8?i6^???????Unknown
?4HostReluGrad",gradient_tape/sequential_9/dense_29/ReluGrad(1      @9      @A      @I      @as?G??8?i3????????Unknown
?5HostMatMul",gradient_tape/sequential_9/dense_30/MatMul_1(1      @9      @A      @I      @as?G??8?i0p???????Unknown
?6HostReadVariableOp"+sequential_9/dense_28/MatMul/ReadVariableOp(1      @9      @A      @I      @as?G??8?i-????????Unknown
w7Host_FusedMatMul"sequential_9/dense_30/BiasAdd(1      @9      @A      @I      @as?G??8?i*??R?????Unknown
?8HostReadVariableOp",sequential_9/dense_30/BiasAdd/ReadVariableOp(1      @9      @A      @I      @as?G??8?i'??????Unknown
t9HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a???	0?izfD$?????Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a???	0?i???D?????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a???	0?i ?e?????Unknown
X<HostCast"Cast_3(1       @9       @A       @I       @a???	0?isx+??????Unknown
X=HostEqual"Equal(1       @9       @A       @I       @a???	0?i??ͦ?????Unknown
|>HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a???	0?i/p??????Unknown
j?HostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a???	0?il???????Unknown
r@HostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a???	0?i????????Unknown
vAHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a???	0?iAW)?????Unknown
vBHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a???	0?ie??I?????Unknown
}CHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a???	0?i???j?????Unknown
yDHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a???	0?iS>??????Unknown
?EHostReluGrad",gradient_tape/sequential_9/dense_28/ReluGrad(1       @9       @A       @I       @a???	0?i^????????Unknown
?FHostReadVariableOp"+sequential_9/dense_30/MatMul/ReadVariableOp(1       @9       @A       @I       @a???	0?i?	???????Unknown
XGHostCast"Cast_4(1      ??9      ??A      ??I      ??a???	 ?i[7?\?????Unknown
aHHostIdentity"Identity(1      ??9      ??A      ??I      ??a???	 ?ie%??????Unknown?
TIHostMul"Mul(1      ??9      ??A      ??I      ??a???	 ?i??v}?????Unknown
vJHostMul"%binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a???	 ?iY???????Unknown
uKHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a???	 ?i???????Unknown
wLHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a???	 ?i?j.?????Unknown
?MHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a???	 ?iWI???????Unknown
?NHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a???	 ?iwO?????Unknown
?OHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a???	 ?i??]??????Unknown
?PHostReadVariableOp",sequential_9/dense_29/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a???	 ?iUҮo?????Unknown
?QHostReadVariableOp"+sequential_9/dense_29/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a???	 ?i?????????Unknown
iRHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
YSHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown*?O
uHostFlushSummaryWriter"FlushSummaryWriter(1     Ѓ@9     Ѓ@A     Ѓ@I     Ѓ@a?d??ز??i?d??ز???Unknown?
`HostGatherV2"
GatherV2_1(1      @@9      @@A      @@I      @@aXg?o3??i7??t????Unknown
dHostDataset"Iterator::Model(1      :@9      :@A      :@I      :@a??.?ʉ??i?ok?h???Unknown
iHostWriteSummary"WriteSummary(1      9@9      9@A      9@I      9@a?`^0/???i?b???%???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A      2@I      2@a"Z??????i???Zk????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      1@9      1@A      1@I      1@a???IS??i??%??-???Unknown
tHost_FusedMatMul"sequential_9/dense_28/Relu(1      1@9      1@A      1@I      1@a???IS??iRt? ????Unknown
?HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      .@9      .@A      .@I      .@a?@q?8P??i ?qa???Unknown
?	HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      *@9      *@A      *@I      *@a??.?ʉ??i?Ҫ??????Unknown
t
Host_FusedMatMul"sequential_9/dense_29/Relu(1      *@9      *@A      *@I      *@a??.?ʉ??i??_ǯ????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      (@9      (@A      (@I      (@a?͍?????i??-J>???Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_30/MatMul(1      &@9      &@A      &@I      &@a???\Ä?i?x?W????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a??K?%???i?? ?????Unknown
wHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      $@9      $@A      $@I      $@a??K?%???i???X(???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      ,@9      ,@A       @I       @aXg?o3~?ie?K??d???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @aXg?o3~?i4?u&????Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_29/MatMul(1       @9       @A       @I       @aXg?o3~?iC?T?????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @amЦmz?i8?Xg???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @amЦmz?im?O[AG???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @amЦmz?i?#?^|???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @amЦmz?i???a?????Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_28/MatMul(1      @9      @A      @I      @amЦmz?id8e?????Unknown
?HostMatMul",gradient_tape/sequential_9/dense_29/MatMul_1(1      @9      @A      @I      @amЦmz?iA?h????Unknown
?HostBiasAddGrad"7gradient_tape/sequential_9/dense_30/BiasAdd/BiasAddGrad(1      @9      @A      @I      @amЦmz?iv??k?O???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @a?͍???v?i?:??|???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?͍???v?i?ۡ?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a??K?%?r?i?r"?????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a??K?%?r?i?	?Q?????Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a??K?%?r?i??#?^???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a??K?%?r?i?7??A???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_9/dense_29/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??K?%?r?i??$4?f???Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aXg?o3n?iᾣ????Unknown
V!HostMean"Mean(1      @9      @A      @I      @aXg?o3n?i?XF????Unknown
V"HostSum"Sum_2(1      @9      @A      @I      @aXg?o3n?i???y????Unknown
?#HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aXg?o3n?iM???????Unknown
z$HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @aXg?o3n?i?*'b?????Unknown
v%HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aXg?o3n?i=?????Unknown
v&HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aXg?o3n?i?O[AG:???Unknown
?'HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aXg?o3n?i?a??zX???Unknown
?(HostBiasAddGrad"7gradient_tape/sequential_9/dense_28/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aXg?o3n?iPt? ?v???Unknown
?)HostReadVariableOp",sequential_9/dense_28/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aXg?o3n?i??)??????Unknown
r*HostSigmoid"sequential_9/dense_30/Sigmoid(1      @9      @A      @I      @aXg?o3n?i???????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?͍???f?i?&w??????Unknown
\,HostGreater"Greater(1      @9      @A      @I      @a?͍???f?i??*'b????Unknown
s-HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?͍???f?i?B޺????Unknown
u.HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?͍???f?iVБN????Unknown
d/HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a?͍???f?i$^E?U$???Unknown
`0HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?͍???f?i???u?:???Unknown
b1HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?͍???f?i?y?	?Q???Unknown
?2HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?͍???f?i?`?Ih???Unknown
?3HostReluGrad",gradient_tape/sequential_9/dense_29/ReluGrad(1      @9      @A      @I      @a?͍???f?i\?1?~???Unknown
?4HostMatMul",gradient_tape/sequential_9/dense_30/MatMul_1(1      @9      @A      @I      @a?͍???f?i*#?Ė????Unknown
?5HostReadVariableOp"+sequential_9/dense_28/MatMul/ReadVariableOp(1      @9      @A      @I      @a?͍???f?i??zX=????Unknown
w6Host_FusedMatMul"sequential_9/dense_30/BiasAdd(1      @9      @A      @I      @a?͍???f?i?>.??????Unknown
?7HostReadVariableOp",sequential_9/dense_30/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?͍???f?i????????Unknown
t8HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @aXg?o3^?i?ծ7?????Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @aXg?o3^?i??{??????Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aXg?o3^?i0?H?????Unknown
X;HostCast"Cast_3(1       @9       @A       @I       @aXg?o3^?id?_????Unknown
X<HostEqual"Equal(1       @9       @A       @I       @aXg?o3^?i???%???Unknown
|=HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aXg?o3^?i???$4???Unknown
j>HostMean"binary_crossentropy/Mean(1       @9       @A       @I       @aXg?o3^?i }?>C???Unknown
r?HostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aXg?o3^?i4J>XR???Unknown
v@HostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aXg?o3^?ih?qa???Unknown
vAHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @aXg?o3^?i?(䭋p???Unknown
}BHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @aXg?o3^?i?1?e????Unknown
yCHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @aXg?o3^?i;~?????Unknown
?DHostReluGrad",gradient_tape/sequential_9/dense_28/ReluGrad(1       @9       @A       @I       @aXg?o3^?i8DK?؝???Unknown
?EHostReadVariableOp"+sequential_9/dense_30/MatMul/ReadVariableOp(1       @9       @A       @I       @aXg?o3^?ilM??????Unknown
XFHostCast"Cast_4(1      ??9      ??A      ??I      ??aXg?o3N?i??h????Unknown
aGHostIdentity"Identity(1      ??9      ??A      ??I      ??aXg?o3N?i?V?D????Unknown?
THHostMul"Mul(1      ??9      ??A      ??I      ??aXg?o3N?i:?? ?????Unknown
vIHostMul"%binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aXg?o3N?i?_??%????Unknown
uJHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??aXg?o3N?in??ز????Unknown
wKHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aXg?o3N?ii??????Unknown
?LHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aXg?o3N?i??e??????Unknown
?MHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aXg?o3N?i<rLlY????Unknown
?NHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aXg?o3N?i??2H?????Unknown
?OHostReadVariableOp",sequential_9/dense_29/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aXg?o3N?ip{$s????Unknown
?PHostReadVariableOp"+sequential_9/dense_29/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aXg?o3N?i     ???Unknown
iQHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i     ???Unknown
YRHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown2CPU