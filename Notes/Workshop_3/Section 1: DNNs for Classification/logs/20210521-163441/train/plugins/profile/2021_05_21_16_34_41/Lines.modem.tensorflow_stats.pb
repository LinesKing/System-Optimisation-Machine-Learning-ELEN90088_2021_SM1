"?d
BHostIDLE"IDLE1     h?@A     h?@a??D??c??i??D??c???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      O@9      O@A      O@I      O@aifb???i?AB4????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      M@9      M@A      M@I      M@a???M5??i\?y#,???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      A@9      A@A      A@I      A@a? ?Ita|?i?%?d???Unknown?
?HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @@9      @@A      @@I      @@aD?'?z?i?t=27????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      F@9      F@A      >@I      >@a?+???
y?iB?
?L????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      :@9      :@A      :@I      :@a7????u?i??u?????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      8@9      8@A      8@I      8@a??R?t?i);??????Unknown
}	HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      3@9      3@A      3@I      3@aq??6?o?i>?b?}????Unknown
?
HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      2@9      2@A      2@I      2@a?,{?n?i?ޝ?]???Unknown
^HostGatherV2"GatherV2(1      1@9      1@A      1@I      1@a? ?Ital?i??'?y???Unknown
`HostGatherV2"
GatherV2_1(1      1@9      1@A      1@I      1@a? ?Ital?ikq?M????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ,@9      @A      ,@I      @a|?"?P_g?i̍&׬????Unknown
vHostExp"%binary_crossentropy/logistic_loss/Exp(1      ,@9      ,@A      ,@I      ,@a|?"?P_g?i}??'????Unknown
dHostDataset"Iterator::Model(1     ?Q@9     ?Q@A      *@I      *@a7????e?i?P_?????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      (@9      (@A      (@I      (@a??R?d?iqn???????Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@aOB? -]b?i?	??%???Unknown?
wHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      &@9      &@A      &@I      &@aOB? -]b?i????????Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      $@9      $@A      $@I      $@a???˱`?i????4$???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?,{?^?iT?6;3???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      "@9      "@A      "@I      "@a?,{?^?iY?\?AB???Unknown
aHostCast"sequential/Cast(1      "@9      "@A      "@I      "@a?,{?^?i??HQ???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a?,{?^?i??vN`???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @aD?'?Z?i?*d??m???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @aD?'?Z?i?>??{???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @aD?'?Z?inR|?_????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a|?"?P_W?i???;????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a|?"?P_W?i u1侟???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a|?"?P_W?iy??n????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a??R?T?iW??r????Unknown
VHostMean"Mean(1      @9      @A      @I      @a??R?T?i5$?w????Unknown
v HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??R?T?i3b{????Unknown
?!HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??R?T?i?A0?????Unknown
\"HostGreater"Greater(1      @9      @A      @I      @a???˱P?iU?'??????Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a???˱P?i?Zu1????Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a???˱P?i?[?????Unknown
~%HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a???˱P?i?sA?????Unknown
?&HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a???˱P?i??'<????Unknown
?'Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a???˱P?iI??????Unknown
o(HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @a???˱P?i???????Unknown
e)Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aD?'?J?i?"?w????Unknown?
?*HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aD?'?J?i?,??H???Unknown
z+HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @aD?'?J?ik6G??!???Unknown
v,HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aD?'?J?iU@?(???Unknown
b-HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aD?'?J?i?JӊQ/???Unknown
?.HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aD?'?J?i)T??5???Unknown
y/HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @aD?'?J?i^_??<???Unknown
?0HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aD?'?J?i?g%ZC???Unknown
?1HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aD?'?J?i?q??J???Unknown
?2HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @aD?'?J?i?{?"?P???Unknown
?3HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aD?'?J?i??w?bW???Unknown
t4HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a??R?D?i*?d\???Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??R?D?i????fa???Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??R?D?i5if???Unknown
X7HostEqual"Equal(1      @9      @A      @I      @a??R?D?iw??5kk???Unknown
u8HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a??R?D?i?*^Ymp???Unknown
V9HostSum"Sum_2(1      @9      @A      @I      @a??R?D?iU??|ou???Unknown
j:HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a??R?D?i?9??qz???Unknown
r;HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a??R?D?i3??s???Unknown
?<HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a??R?D?i?H??u????Unknown
v=HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??R?D?i?Dx????Unknown
v>HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a??R?D?i?W?.z????Unknown
~?HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a??R?D?i??mR|????Unknown
?@HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??R?D?i^fv~????Unknown
?AHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??R?D?i?햙?????Unknown
?BHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a??R?D?i<u+??????Unknown
}CHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      @9      @A      @I      @a??R?D?i?????????Unknown
?DHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a??R?D?i?T?????Unknown
?EHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??R?D?i??'?????Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @aD?'?:?i~L?ߴ???Unknown
vGHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aD?'?:?is??6????Unknown
VHHostCast"Cast(1       @9       @A       @I       @aD?'?:?iho?????Unknown
XIHostCast"Cast_2(1       @9       @A       @I       @aD?'?:?i]u1?????Unknown
XJHostCast"Cast_4(1       @9       @A       @I       @aD?'?:?iR$??:????Unknown
sKHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @aD?'?:?iG);??????Unknown
|LHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aD?'?:?i<.?x?????Unknown
?MHostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1       @9       @A       @I       @aD?'?:?i13;?????Unknown
}NHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @aD?'?:?i&8d??????Unknown
`OHostDivNoNan"
div_no_nan(1       @9       @A       @I       @aD?'?:?i=ǿ?????Unknown
xPHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @aD?'?:?iB*?C????Unknown
?QHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @aD?'?:?iG?D?????Unknown
~RHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @aD?'?:?i?K??????Unknown
?SHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aD?'?:?i?PS?G????Unknown
THostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @aD?'?:?i?U???????Unknown
XUHostCast"Cast_3(1      ??9      ??A      ??I      ??aD?'?*?i^???I????Unknown
aVHostIdentity"Identity(1      ??9      ??A      ??I      ??aD?'?*?i?ZN?????Unknown?
TWHostMul"Mul(1      ??9      ??A      ??I      ??aD?'?*?iR?J??????Unknown
dXHostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??aD?'?*?i?_|L????Unknown
uYHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??aD?'?*?iF??q?????Unknown
wZHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aD?'?*?i?d?Ң????Unknown
?[HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??aD?'?*?i:?4N????Unknown
?\HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aD?'?*?i?iB??????Unknown
?]Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aD?'?*?i.?s??????Unknown
?^HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??aD?'?*?i?n?WP????Unknown
?_HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??aD?'?*?i"?ָ?????Unknown
?`HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aD?'?*?i?s?????Unknown
?aHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aD?'?*?i?9{R????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??aD?'?*?i?xk??????Unknown
?cHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??aD?'?*?i
??=?????Unknown
?dHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??aD?'?*?i?}ΞT????Unknown
?eHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aD?'?*?i?????????Unknown
LfHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown
YgHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown
]hHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(i?????????Unknown*?c
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      O@9      O@A      O@I      O@aVq?K%s??iVq?K%s???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      M@9      M@A      M@I      M@a?a&?\ ??i?????????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      A@9      A@A      A@I      A@a??1R???i??g??????Unknown?
?HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @@9      @@A      @@I      @@a?p??,??i ??[????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      F@9      F@A      >@I      >@a????????i2R??????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      :@9      :@A      :@I      :@ai???/Ԣ?i?????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      8@9      8@A      8@I      8@a׺T2ga??i?be?F???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      3@9      3@A      3@I      3@a?'eㄛ?iWŵ.?????Unknown
?	HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      2@9      2@A      2@I      2@aC???i۶m۶m???Unknown
^
HostGatherV2"GatherV2(1      1@9      1@A      1@I      1@a??1R???if7???????Unknown
`HostGatherV2"
GatherV2_1(1      1@9      1@A      1@I      1@a??1R???i???!?????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ,@9      @A      ,@I      @a??be?F??i???????Unknown
vHostExp"%binary_crossentropy/logistic_loss/Exp(1      ,@9      ,@A      ,@I      ,@a??be?F??i??,@????Unknown
dHostDataset"Iterator::Model(1     ?Q@9     ?Q@A      *@I      *@ai???/Ԓ?i?g??????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      (@9      (@A      (@I      (@a׺T2ga??i???????Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a?V?1=ݏ?i{??a&???Unknown?
wHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      &@9      &@A      &@I      &@a?V?1=ݏ?iv誸֥???Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      $@9      $@A      $@I      $@af7??????iT?h????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aC???i????????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      "@9      "@A      "@I      "@aC???i?F????Unknown
aHostCast"sequential/Cast(1      "@9      "@A      "@I      "@aC???iw/??R???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@aC???i?]׺???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?p??,??i?Ҿ;????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a?p??,??i?? b;t???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @a?p??,??i?Z???????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??be?F??i??j	"???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a??be?F??iTq?K%s???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a??be?F??i??B-A????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a׺T2ga??i?O??	???Unknown
VHostMean"Mean(1      @9      @A      @I      @a׺T2ga??i???fLO???Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a׺T2ga??i}??Ҕ???Unknown
? HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a׺T2ga??ihHh?W????Unknown
\!HostGreater"Greater(1      @9      @A      @I      @af7????|?i?be?F???Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @af7????|?iF}bP6N???Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @af7????|?i??_?%????Unknown
~$HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @af7????|?i$?\ ????Unknown
?%HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @af7????|?i??YX????Unknown
?&Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @af7????|?i?V??5???Unknown
o'HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @af7????|?iqT?o???Unknown
e(Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?p??,w?ic??<????Unknown?
?)HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?p??,w?iUŵ.?????Unknown
z*HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?p??,w?iG??A?????Unknown
v+HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?p??,w?i9?UG)???Unknown
b,HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?p??,w?i+kHh?W???Unknown
?-HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?p??,w?iMy{?????Unknown
y.HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a?p??,w?i/??R????Unknown
?/HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?p??,w?iۡ?????Unknown
?0HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?p??,w?i??????Unknown
?1HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a?p??,w?i??<?]????Unknown
?2HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?p??,w?i׶m۶m???Unknown
t3HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a׺T2gaq?iM`ҩy????Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a׺T2gaq?i?	7x<????Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a׺T2gaq?i9??F?????Unknown
X6HostEqual"Equal(1      @9      @A      @I      @a׺T2gaq?i?\ ?????Unknown
u7HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a׺T2gaq?i%e?????Unknown
V8HostSum"Sum_2(1      @9      @A      @I      @a׺T2gaq?i??ɱG>???Unknown
j9HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a׺T2gaq?iY.?
a???Unknown
r:HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a׺T2gaq?i??N̓???Unknown
?;HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a׺T2gaq?i????????Unknown
v<HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a׺T2gaq?isU\?R????Unknown
v=HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a׺T2gaq?i????????Unknown
~>HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a׺T2gaq?i_?%?????Unknown
??HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a׺T2gaq?i?Q?V?1???Unknown
?@HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a׺T2gaq?iK??$^T???Unknown
?AHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a׺T2gaq?i??S? w???Unknown
}BHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      @9      @A      @I      @a׺T2gaq?i7N???????Unknown
?CHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a׺T2gaq?i????????Unknown
?DHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a׺T2gaq?i#??^i????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?p??,g?i??????Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?p??,g?i??q????Unknown
VGHostCast"Cast(1       @9       @A       @I       @a?p??,g?i?J??$???Unknown
XHHostCast"Cast_2(1       @9       @A       @I       @a?p??,g?ie??<???Unknown
XIHostCast"Cast_4(1       @9       @A       @I       @a?p??,g?i ?{HS???Unknown
sJHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?p??,g?i?F?tj???Unknown
|KHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?p??,g?i???!?????Unknown
?LHostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1       @9       @A       @I       @a?p??,g?i?(E?͘???Unknown
}MHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?p??,g?i???4?????Unknown
`NHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?p??,g?i?
v?&????Unknown
xOHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?p??,g?i?{HS????Unknown
?PHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?p??,g?i????????Unknown
~QHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?p??,g?i?]?[????Unknown
?RHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?p??,g?i?????#???Unknown
SHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a?p??,g?i??pn;???Unknown
XTHostCast"Cast_3(1      ??9      ??A      ??I      ??a?p??,W?i7x<??F???Unknown
aUHostIdentity"Identity(1      ??9      ??A      ??I      ??a?p??,W?i???1R???Unknown?
TVHostMul"Mul(1      ??9      ??A      ??I      ??a?p??,W?i1??<?]???Unknown
dWHostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??a?p??,W?i?!??^i???Unknown
uXHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?p??,W?i+Zm??t???Unknown
wYHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?p??,W?i??9?????Unknown
?ZHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?p??,W?i%?P!????Unknown
?[HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?p??,W?i?Ҕ?????Unknown
?\Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?p??,W?i<??M????Unknown
?]HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?p??,W?i?tj?????Unknown
?^HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a?p??,W?i?6cz????Unknown
?_HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?p??,W?i???????Unknown
?`HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?p??,W?i???????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?p??,W?i?V?1=????Unknown
?bHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a?p??,W?i?gv?????Unknown
?cHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?p??,W?i??3?i????Unknown
?dHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?p??,W?i     ???Unknown
LeHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i     ???Unknown
YfHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown
]gHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(i     ???Unknown2CPU