
public class DixonIndFactory implements IndFactory {

    private final int n;

    public DixonIndFactory(int n) {
        this.n = n;
    }

    @Override
    public DixonInd getInstance() {
        return new DixonInd(this.n);
    }
}
